import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import {type TextSplitterConfig} from './config.ts'
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';

export class DocumentProcessor {
    private pdfPath: string;
    private textSplitterConfig: TextSplitterConfig;

  constructor(pdfPath: string, config: TextSplitterConfig) {
    this.pdfPath = pdfPath;
    this.textSplitterConfig = config;
  }
  
  async loadAndSplit() {
    const loader = new PDFLoader(this.pdfPath);
    const documents = await loader.load();
    const splitter = new RecursiveCharacterTextSplitter(this.textSplitterConfig);
    const chunks = await splitter.splitDocuments(documents);

    return chunks;
  }

}