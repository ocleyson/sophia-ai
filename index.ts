import { ChatOpenAI } from "langchain/chat_models/openai";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import 'dotenv/config'
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import {
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
} from "langchain/prompts";
import {
    RunnableSequence,
} from "langchain/schema/runnable";
import { formatDocumentsAsString } from "langchain/util/document";
import readline from 'node:readline/promises';
import * as fs from "fs";
import { stdin as input, stdout as output } from 'node:process';
import { BufferMemory } from "langchain/memory";
import { Document } from "langchain/document";
import { LLMChain } from "langchain/chains";
import chalk from "chalk";
import { BaseMessage } from "langchain/schema";

import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import {
  JSONLoader,
  JSONLinesLoader,
} from "langchain/document_loaders/fs/json";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { CSVLoader } from "langchain/document_loaders/fs/csv";

const memory = new BufferMemory({
  memoryKey: "chatHistory",
  inputKey: "question",
  outputKey: "text",
  returnMessages: true,
});

const serializeChatHistory = (chatHistory: Array<BaseMessage>): string =>
    chatHistory
    .map((chatMessage) => {
      if (chatMessage._getType() === "human") {
        return `Human: ${chatMessage.content}`;
      } else if (chatMessage._getType() === "ai") {
        return `Assistant: ${chatMessage.content}`;
      } else {
        return `${chatMessage.content}`;
      }
    })
    .join("\n");

async function initializeChat() {
    console.log(chalk.red('Carregando dados...'));

    const model = new ChatOpenAI({
        openAIApiKey: process.env.OPENAI_API_KEY,
        modelName: 'gpt-3.5-turbo',
    });

    const directoryLoader = new DirectoryLoader(
      "data/knowledge",
      {
        ".json": (path) => new JSONLoader(path, "/texts"),
        ".jsonl": (path) => new JSONLinesLoader(path, "/html"),
        ".txt": (path) => new TextLoader(path),
        ".csv": (path) => new CSVLoader(path, "text"),
        ".pdf": (path) => new PDFLoader(path),
      }
    );

    const directoryOutput = await directoryLoader.load();

    const splitter = new RecursiveCharacterTextSplitter();

    const docs = await splitter.splitDocuments(directoryOutput);

    const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());

    const vectorStoreRetriever = vectorStore.asRetriever();

    const SYSTEM_TEMPLATE = `Use as seguintes partes do contexto para responder à pergunta no final.
    Se você não sabe a resposta, apenas diga que não sabe, não tente inventar uma resposta.
    ----------------
    CONTEXTO: {context}
    ----------------
    HISTÓRICO: {chatHistory}
    `;

    const messages = [
        SystemMessagePromptTemplate.fromTemplate(SYSTEM_TEMPLATE),
        HumanMessagePromptTemplate.fromTemplate("{question}"),
    ];

    const prompt = ChatPromptTemplate.fromMessages(messages);

    const slowerChain = new LLMChain({
        llm: model,
        prompt,
    });

    const performQuestionAnswering = async (input: {
        question: string;
        chatHistory: Array<BaseMessage> | null;
        context: Array<Document>;
      }): Promise<{ result: string; sourceDocuments: Array<Document> }> => {
        let newQuestion = input.question;
        
        const serializedDocs = formatDocumentsAsString(input.context);
        const chatHistoryString = input.chatHistory
          ? serializeChatHistory(input.chatHistory)
          : null;
      
        const response = await slowerChain.invoke({
          chatHistory: chatHistoryString ?? "",
          context: serializedDocs,
          question: newQuestion,
        });
      
        await memory.saveContext(
          {
            question: input.question,
          },
          {
            text: response.text,
          }
        );
      
        return {
          result: response.text,
          sourceDocuments: input.context,
        };
      };

    const chain = RunnableSequence.from([
        {
            question: (input: { question: string; }) => input.question,
            chatHistory: async () => {
                const savedMemory = await memory.loadMemoryVariables({});
                const hasHistory = savedMemory.chatHistory.length > 0;
                return hasHistory ? savedMemory.chatHistory : null;
            },
            context: async (input: { question: string }) => vectorStoreRetriever.getRelevantDocuments(input.question),
        },
        performQuestionAnswering,
    ]);

    console.log(chalk.red('Dados carregados!'));

    return chain;
}

async function handleUserInput(input: string, customChain: RunnableSequence<{ question: string }, { result: string; sourceDocuments: Array<Document>; }>) {
    let answer;
    let question;

    try {
        question = fs.readFileSync(`data/questions/${input}.txt`, "utf8");
    } catch (e) {
        question = input;
    }

    try {
      answer = await customChain.invoke({
        question,
      });
  
      console.log(chalk.green('Resposta: ', answer.result));
    } catch (e) {   
      console.log(chalk.red('Error: ', e));
    }
}

async function startChat() {
    const rl = readline.createInterface({
        input,
        output,
    });

    const customChain = await initializeChat();

    async function getUserQuestion() {
        const questionsFileName = await rl.question('Enter the question or file name: ');

        if (questionsFileName.toLowerCase() === 'exit') {
            await memory.clear();
            rl.close();
            console.log(chalk.red('Chat closed!'))
        } else if (questionsFileName.toLowerCase() === 'clearmemory') {
            await memory.clear();
            console.log(chalk.red('Memory cleared!'))
            getUserQuestion();
        } else {
            await handleUserInput(questionsFileName, customChain);
            getUserQuestion();
        }
    }

    getUserQuestion();
}

startChat();
