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
import { BaseMessage } from "langchain/dist/schema";
import { Document } from "langchain/document";
import { LLMChain } from "langchain/chains";

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

async function initializeChat(dataFile: string) {
    console.log('Carregando dados...')

    const memory = new BufferMemory({
        memoryKey: "chatHistory",
        inputKey: "question",
        outputKey: "text",
        returnMessages: true,
      });

    const model = new ChatOpenAI({
        openAIApiKey: process.env.OPENAI_API_KEY,
    });

    const pdfLoader = new PDFLoader(`data/knowledge/${dataFile}.pdf`);

    const pdfOutput = await pdfLoader.load();

    const splitter = new RecursiveCharacterTextSplitter();

    const pdfSplit = await splitter.splitDocuments(pdfOutput);

    const vectorStore = await HNSWLib.fromDocuments(pdfSplit, new OpenAIEmbeddings());

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

    console.log('Dados carregados!')

    return chain;
}

async function handleUserInput(input: string, customChain: RunnableSequence<{ question: string }, { result: string; sourceDocuments: Array<Document>; }>) {
    let answer;
    let question;

    try {
        question = fs.readFileSync(`data/questions/${input}.txt`, "utf8");
        answer = await customChain.invoke({
            question,
        });
    } catch (e) {
        question = input;
        answer = await customChain.invoke({
            question,
        });
    }

    console.log('Resposta: ', answer.result);
}

async function startChat() {
    const rl = readline.createInterface({
        input,
        output,
    });

    const dataFileName = await rl.question('Enter data file name: ');

    const customChain = await initializeChat(dataFileName);

    async function getUserQuestion() {
        const questionsFileName = await rl.question('Enter the question or file name: ');

        if (questionsFileName.toLowerCase() === 'exit') {
            rl.close();
        } else {
            await handleUserInput(questionsFileName, customChain);
            getUserQuestion();
        }
    }

    getUserQuestion();
}

startChat();
