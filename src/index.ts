import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { ChatOpenAI } from "@langchain/openai";
import { CONFIG } from "./config.ts";
import { DocumentProcessor } from "./documentProcessor.ts";
import { type PretrainedOptions } from "@huggingface/transformers";
import neo4j from "neo4j-driver";

const driver = neo4j.driver(
    CONFIG.neo4j.url,
    neo4j.auth.basic(CONFIG.neo4j.username, CONFIG.neo4j.password)
);

const storeEmbeddingsInNeo4j = async (
    chunks: Array<{ pageContent: string; metadata: Record<string, unknown> }>,
    embeddings: number[][]
) => {
    const session = driver.session();

    try {
        for (let i = 0; i < chunks.length; i++) {
            const chunk = chunks[i]!;
            const embedding = embeddings[i]!;

            await session.run(
                `CREATE (n:${CONFIG.neo4j.nodeLabel} {
                    text: $text,
                    page: $page,
                    embedding: $embedding,
                    createdAt: datetime()
                })`,
                {
                    text: chunk.pageContent,
                    page: (chunk.metadata as any)?.loc?.pageNumber || 0,
                    embedding: embedding,
                }
            );
        }

        await session.run(
            `CALL db.index.vector.createOrUpdate(
                $indexName,
                "COSINE",
                {nodeLabel: $nodeLabel, propertyKey: "embedding", dimensions: 384}
            )`,
            {
                indexName: CONFIG.neo4j.indexName,
                nodeLabel: CONFIG.neo4j.nodeLabel,
            }
        );

        console.log(`✅ ${embeddings.length} chunks armazenados no Neo4j`);
    } finally {
        await session.close();
    }
};

const semanticSearch = async (query: string, embeddingModel: HuggingFaceTransformersEmbeddings) => {
    const session = driver.session();

    try {
        const queryEmbedding = await embeddingModel.embedQuery(query);

        const results = await session.run(
            `CALL db.index.vector.queryNodes($indexName, $k, $queryEmbedding)
             YIELD node, score
             RETURN node.text as text, score
             ORDER BY score DESC`,
            {
                indexName: CONFIG.neo4j.indexName,
                k: CONFIG.similarity.topK,
                queryEmbedding: queryEmbedding,
            }
        );

        return results.records.map(record => ({
            text: record.get("text"),
            score: record.get("score"),
        }));
    } finally {
        await session.close();
    }
};

const generateAnswer = async (query: string, context: Array<{ text: string; score: number }>) => {
    const llm = new ChatOpenAI({
        apiKey: CONFIG.openRouter.apiKey,
        basePath: CONFIG.openRouter.url,
        model: CONFIG.openRouter.nlpModel || "google/gemma-3-27b-it:free",
        temperature: CONFIG.openRouter.temperature,
        defaultHeaders: CONFIG.openRouter.defaultHeaders,
    } as any);

    const contextText = context.map(c => c.text).join("\n---\n");
    const prompt = `Baseado no contexto abaixo, responda a pergunta:

CONTEXTO:
${contextText}

PERGUNTA: ${query}

Forneça uma resposta clara e precisa baseada apenas no contexto fornecido.`;

    const message = await llm.invoke(prompt);
    return message.content;
};

try {
    console.log("🚀 Iniciando pipeline de embeddings...");

    const documentProcessor = new DocumentProcessor(CONFIG.pdf.path, CONFIG.textSplitter);

    const embeddingModel = new HuggingFaceTransformersEmbeddings({
        model: CONFIG.embedding.modelName,
        pretrainedOptions: CONFIG.embedding.pretrainedOptions as PretrainedOptions,
    });

    const chunks = await documentProcessor.loadAndSplit();
    console.log(`📄 ${chunks.length} chunks extraídos do PDF`);

    const embeddings = await embeddingModel.embedDocuments(chunks.map(chunk => chunk.pageContent));
    console.log(`🧠 Embeddings gerados (dimensões: ${embeddings[0]?.length || 0})`);

    await storeEmbeddingsInNeo4j(chunks, embeddings);

    // Exemplo de consulta semântica
    const testQuery = "O que é um tensor?";
    console.log(`\n🔍 Buscando: "${testQuery}"`);

    const searchResults = await semanticSearch(testQuery, embeddingModel);
    console.log(`\n📊 Top ${CONFIG.similarity.topK} resultados encontrados:`);
    searchResults.forEach((result, i) => {
        console.log(`${i + 1}. (score: ${(result.score * 100).toFixed(2)}%)`);
        console.log(`   ${result.text.substring(0, 100)}...`);
    });

    const answer = await generateAnswer(testQuery, searchResults);
    console.log(`\n💬 Resposta gerada:\n${answer}`);

    await driver.close();
} catch (error) {
    console.error("❌ Erro:", error);
    await driver.close();
}
