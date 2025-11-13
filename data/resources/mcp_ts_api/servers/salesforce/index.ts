/**
 * Salesforce MCP Server TypeScript API
 *
 * Provides TypeScript API wrappers for Salesforce MCP tools.
 * Large data stays in sandbox; only summaries/logs returned.
 */
import { callMCPTool } from '../../callMCPTool';

export interface UpdateRecordInput {
  objectType: string;
  recordId: string;
  data: Record<string, any>;
}

export interface UpdateRecordOutput {
  success: boolean;
  recordId: string;
}

export interface QueryRecordsInput {
  objectType: string;
  where?: string;
  limit?: number;
}

export interface QueryRecordsOutput {
  records: Array<{
    Id: string;
    [key: string]: any;
  }>;
}

/**
 * Update a Salesforce record.
 * Large data processed in sandbox; only success status returned.
 */
export async function updateRecord(
  input: UpdateRecordInput
): Promise<UpdateRecordOutput> {
  return callMCPTool<UpdateRecordInput, UpdateRecordOutput>(
    'salesforce__update_record',
    input
  );
}

/**
 * Query Salesforce records.
 * Returns record metadata; large fields stay in sandbox.
 */
export async function queryRecords(
  input: QueryRecordsInput
): Promise<QueryRecordsOutput> {
  return callMCPTool<QueryRecordsInput, QueryRecordsOutput>(
    'salesforce__query_records',
    input
  );
}

