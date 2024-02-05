const fs = require("fs");
const readline = require("readline");
const pdfParse = require("pdf-parse");
const { CharacterTextSplitter } = require("langchain/text_splitter");
const { OpenAIEmbeddings, OpenAI } = require("@langchain/openai");
const { RetrievalQAChain } = require("langchain/chains");
const { PromptTemplate } = require("langchain/prompts");
const { MemoryVectorStore } = require("langchain/vectorstores/memory");

require("dotenv").config();

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

const askQuestion = (query) => {
  return new Promise((resolve) =>
    rl.question(query, (answer) => resolve(answer))
  );
};

const readPDF = async (filename) => {
  try {
    const dataBuffer = fs.readFileSync(filename);
    const data = await pdfParse(dataBuffer);
    return data.text;
  } catch (error) {
    console.error("Error reading PDF:", error);
    return null;
  }
};

const main = async () => {
  const filename = await askQuestion("Enter the name of the PDF file: ");
  const pdf = await readPDF(filename);

  // split into chunks
  const splitter = new CharacterTextSplitter({
    separator: "\n",
    chunkSize: 1000,
    chunkOverlap: 200,
  });

  const text = await splitter.createDocuments([pdf]);

  // console.log(text);

  const embeddings = new OpenAIEmbeddings();

  // console.log(embeddings);
  const vectorStore = await MemoryVectorStore.fromDocuments(text, embeddings);
  // console.log(vectorStore);

  const llm = new OpenAI({
    modelName: "gpt-3.5-turbo", // Defaults to "gpt-3.5-turbo-instruct" if no model provided.
  });

  const template = `Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
{context}
Question: {question}
Helpful Answer:`;

  const chain = RetrievalQAChain.fromLLM(llm, vectorStore.asRetriever(), {
    prompt: PromptTemplate.fromTemplate(template),
    returnSourceDocuments: true,
  });

  if (pdf) {
    let exit = false;
    while (!exit) {
      const question = await askQuestion(
        "Ask a question about the PDF (or type 'exit' to quit): "
      );
      if (question.toLowerCase() === "exit") {
        exit = true;
      } else {
        const response = await chain.invoke({
          query: question,
        });

        console.log(response.text);
        // console.log(JSON.stringify(response.sourceDocuments[0].pageContent));
      }
    }
  }
  // What is the AWS Well-Architected Framework?
  rl.close();
};

main();
