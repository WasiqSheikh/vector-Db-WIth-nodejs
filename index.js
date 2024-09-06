import { Pinecone } from "@pinecone-database/pinecone";
import dotenv from 'dotenv';
import { HfInference } from "@huggingface/inference";
import express from "express";
import uuid4 from "uuid4";
import fs from 'fs';
import { Buffer } from "buffer";
import cors from 'cors';
dotenv.config();
const pc = new Pinecone({ apiKey: process.env.PINECONE_API });
const app = express();
app.use(cors());
app.use(express.json());
const indexName = "vectordb-crud";
const indexes = await pc.listIndexes();
const indexExist = indexes.indexes.find((index) => index.name === indexName);

if (!indexExist) {
    await pc.createIndex({
        name: indexName,
        dimension: 384,
        metric: 'cosine',
        spec: {
            serverless: {
                cloud: 'aws',
                region: 'us-east-1'
            }
        }
    });

    console.log('index created successfully');
}
else {
    console.log('index already exist');
}

const hf_token = process.env.HF_TOKEN;

app.post('/create-record', async (req, res) => {
    try {
        const { title, description } = req.body;
        if (!title || !description) {
            return res.status(400).json({ error: 'title and description are required' });
        }

        const response = await createRecord(indexName, title, description);
        res.status(201).json({ message: 'Record created successfully', response });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.get('/get-all-records', async (req, res) => {
    const articles = [];
    try {
        const records = await getAllRecords(indexName);
        records.matches.forEach((match) => {
            articles.push({
                id: match.id,
                title: match.metadata.title,
                description: match.metadata.description
            });
        });
        res.status(200).json(articles);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
})

app.get('/get-record/:id', async (req, res) => {
    try {
        const { id } = req.params;
        const record = await getRecord(indexName, id);
        res.status(200).json({ record });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
})

app.post('/update-record/:id', async (req, res) => {
    try {
        const { id } = req.params;
        const { title, description } = req.body;

        const update = await updateRecord(indexName, id, title, description);
        res.status(200).json({ message: 'Record updated successfully' });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
})

app.delete('/delete-record/:id', async (req, res) => {
    try {
        const { id } = req.params;
        await deleteRecord(indexName, id);
        res.status(200).json({ message: 'Record deleted successfully' });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
})

app.get('/search-record/:query', async (req, res) => {
    try {
        const { query } = req.params;
        const result = await searchRecord(indexName, query);
        res.status(200).json({ result });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
})

app.get('/summarize/:id', async (req, res) => {
    try {
        const { id } = req.params;
        const record = await getRecord(indexName, id);
        const result = await summarize(id, record);
        res.status(200).json({ result });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
})

app.get('/convert-text-to-speech/:id', async (req, res) => {
    try {
        const { id } = req.params;
        const record = await getRecord(indexName, id);
        await convertTextToSpeech(id, record);
        const __dirname = process.cwd();
        const filename = __dirname + '/' + 'generated_audio.mp3';
        res.status(200).sendFile(filename);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
})

app.get('/convert-text-to-image/:id', async (req, res) => {
    try {
        const { id } = req.params;
        console.log('id = ', id);
        const record = await getRecord(indexName, id);
        const blob = await generateImage(id, record);
        // const __dirname = process.cwd();
        // const filename = __dirname + '/' + 'generated_image_from.jpg';
        res.status(200).send({blob: blob});
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
})

app.get('/feature-extract/:id', async (req, res) => {
    try {
        const { id } = req.params;
        const record = await getRecord(indexName, id);
        const features = await ExtractFeatures(id, record);
        res.status(200).json({ features });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
})

async function createRecord(indexName, title, description) {
    const index = pc.Index(indexName);
    const id = uuid4();
    const concatenatedString = title.concat(description);
    const result = await createEmbedding(concatenatedString);
    const upsertResponse = await index.namespace("vDB").upsert([
        {
            id,
            values: result,
            metadata: {
                title,
                description
            }
        },
    ]);
    return upsertResponse;
}

async function getAllRecords(indexName) {
    const index = pc.Index(indexName);
    const vector = await createEmbedding('');
    const result = await index.namespace("vDB").query({
        vector: vector,
        topK: 10,
        includeValues: false,
        includeMetadata: true
    })
    return result;
}

async function getRecord(indexName, id) {
    const index = pc.Index(indexName);
    const record = await index.namespace("vDB").fetch([id]);
    return record;
}

async function updateRecord(indexName, id, title, description) {
    const index = pc.Index(indexName);
    const concatenatedString = title.concat(description);
    const result = await createEmbedding(concatenatedString);
    const upsertResponse = await index.namespace("vDB").update({
        id,
        values: result,
        metadata: {
            title,
            description
        }
    });
    return upsertResponse;
}

async function deleteRecord(indexName, id) {
    const index = pc.Index(indexName);
    const vector = await index.namespace("vDB").deleteOne(id);
    return vector;
}

async function searchRecord(indexName, query) {
    const index = pc.Index(indexName);
    const vector = await createEmbedding(query);
    const result = await index.namespace("vDB").query({
        vector: vector,
        topK: 10,
        includeValues: false,
        includeMetadata: true
    });

    console.log('result = ', result);
    return result;
}

async function createEmbedding(text) {
    const hf = new HfInference(hf_token);
    const model = 'sentence-transformers/all-MiniLM-L6-v2';
    try {
        const result = await hf.featureExtraction({
            model,
            inputs: text,
        });
        return result;
    } catch (err) {
        console.error(err);
    }
}

async function summarize(id, record) {
    const hf = new HfInference(hf_token);
    const model = 'facebook/bart-large-cnn';
    try {
        const result = await hf.summarization({
            model,
            inputs: record.records[id].metadata.description,
            parameters: {
                max_length: 40
            }
        });
        return result;
    } catch (err) {
        console.error(err);
    }
}

async function convertTextToSpeech(id, record) {
    const hf = new HfInference(hf_token);
    //const model = 'openai/whisper-tiny';
    const model = 'espnet/kan-bayashi_ljspeech_vits';
    try {
        const response = await hf.textToSpeech({
            model,
            inputs: record.records[id].metadata.description
        });
        if (response) {
            console.log('response = ', response)
            const buffer = await response.arrayBuffer();
            const imageData = Buffer.from(buffer);
            const filename = 'generated_audio.mp3';
            fs.writeFileSync(filename, imageData);

            return `Image saved to: ${filename}`;
        } else {
            return null;
        }
    } catch (err) {
        console.error(err);
    }
}


async function generateImage(id, record) {
    const hf = new HfInference(hf_token);
    const model = "stabilityai/stable-diffusion-3-medium-diffusers";
    const response = await hf.textToImage({
      inputs: record.records[id].metadata.description,
      model: model,
    });

    console.log('response = ', response)

    if (response) {
        const buffer = await response.arrayBuffer();
        const imageData = Buffer.from(buffer);
        console.log('imageData = ', imageData);
        return imageData;
        const filename = 'generated_image_from_text.jpg';
        fs.writeFileSync(filename, imageData);

        return `Image saved to: ${filename}`;
      } else {
        return null;
      }

}

async function ExtractFeatures(id, record) {
    const hf = new HfInference(hf_token);
    //const model = "sampathkethineedi/industry-classification-api";
    /** Required */
    const model = "sampathkethineedi/industry-classification-api";
    /** Required */
    
    //const model = 'ashok2216/gpt2-amazon-sentiment-classifier-V1.0';
    
    //const model = 'Maha/hin-trac2';
    try {
        const response = await hf.textClassification({
            model,
            inputs: record.records[id].metadata.description
        });
        console.log('response = ', response);
        return response;
    } catch (err) {
        console.error(err);
    }
}

app.listen(3000, () => {
    console.log('Server is running on port 3000');
});