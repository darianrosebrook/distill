/**
 * Google Drive MCP Server TypeScript API
 *
 * Provides TypeScript API wrappers for Google Drive MCP tools.
 * Large data stays in sandbox; only summaries/logs returned.
 */
import { callMCPTool } from '../../callMCPTool';

export interface GetDocumentInput {
  documentId: string;
}

export interface GetDocumentOutput {
  content: string;
  title: string;
  id: string;
}

export interface ListFilesInput {
  folderId?: string;
  limit?: number;
}

export interface ListFilesOutput {
  items: Array<{
    id: string;
    name: string;
    mimeType: string;
  }>;
}

/**
 * Get a document by ID.
 * Large content stays in sandbox; only metadata returned in tokens.
 */
export async function getDocument(
  input: GetDocumentInput
): Promise<GetDocumentOutput> {
  return callMCPTool<GetDocumentInput, GetDocumentOutput>(
    'google_drive__get_document',
    input
  );
}

/**
 * List files in a folder.
 * Returns file metadata only; content not included.
 */
export async function listFiles(
  input: ListFilesInput = {}
): Promise<ListFilesOutput> {
  return callMCPTool<ListFilesInput, ListFilesOutput>(
    'google_drive__list_files',
    input
  );
}











