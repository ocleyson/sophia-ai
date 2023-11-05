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
    RunnablePassthrough,
    RunnableSequence,
} from "langchain/schema/runnable";
import { formatDocumentsAsString } from "langchain/util/document";
import { StringOutputParser } from "langchain/schema/output_parser";
import readline from 'node:readline/promises';
import * as fs from "fs";
import { stdin as input, stdout as output } from 'node:process';

async function initializeChat(dataFile: string) {
    console.log('Carregando dados...')

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
    {context}`;

    const messages = [
        SystemMessagePromptTemplate.fromTemplate(SYSTEM_TEMPLATE),
        HumanMessagePromptTemplate.fromTemplate("{question}"),
    ];

    const prompt = ChatPromptTemplate.fromMessages(messages);

    const chain = RunnableSequence.from([
        {
            context: vectorStoreRetriever.pipe(formatDocumentsAsString),
            question: new RunnablePassthrough(),
        },
        prompt,
        model,
        new StringOutputParser(),
    ]);

    console.log('Dados carregados!')

    return chain;
}

async function handleUserInput(input: string, customChain: RunnableSequence<any, string>) {

    try {
        const text = fs.readFileSync(`data/questions/${input}.txt`, "utf8");

        const answer = await customChain.invoke(text);

        console.log('Resposta: ', answer);
    } catch (e) {
        console.log('error: ', e);
    }
}

async function startChat() {
    const rl = readline.createInterface({
        input,
        output,
    });

    const dataFileName = await rl.question('Enter data file name: ');

    const customChain = await initializeChat(dataFileName);

    async function getUserQuestion() {
        const questionsFileName = await rl.question('Enter questions txt file name: ');

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