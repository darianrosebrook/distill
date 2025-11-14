/**
 * Slack MCP Server TypeScript API
 *
 * Provides TypeScript API wrappers for Slack MCP tools.
 * Large data stays in sandbox; only summaries/logs returned.
 */
import { callMCPTool } from '../../callMCPTool';

export interface SendMessageInput {
  channel: string;
  text: string;
  threadTs?: string;
}

export interface SendMessageOutput {
  success: boolean;
  ts: string;
}

export interface GetChannelsInput {
  types?: string[];
  excludeArchived?: boolean;
}

export interface GetChannelsOutput {
  channels: Array<{
    id: string;
    name: string;
    isPrivate: boolean;
  }>;
}

/**
 * Send a message to a Slack channel.
 * Large attachments stay in sandbox; only message ID returned.
 */
export async function sendMessage(
  input: SendMessageInput
): Promise<SendMessageOutput> {
  return callMCPTool<SendMessageInput, SendMessageOutput>(
    'slack__send_message',
    input
  );
}

/**
 * Get list of Slack channels.
 * Returns channel metadata only; message content not included.
 */
export async function getChannels(
  input: GetChannelsInput = {}
): Promise<GetChannelsOutput> {
  return callMCPTool<GetChannelsInput, GetChannelsOutput>(
    'slack__get_channels',
    input
  );
}










