// Package gemini is the llm.Provider for Google's Gemini API, built on
// google.golang.org/genai. No genai type crosses the package boundary.
//
// # Client
//
// New always configures the Gemini API backend with the given API key and
// an explicit base URL (https://generativelanguage.googleapis.com/ unless
// WithBaseURL says otherwise). That stops genai from switching to Vertex AI
// via GOOGLE_GENAI_USE_VERTEXAI / GOOGLE_GENAI_USE_ENTERPRISE, from using
// GOOGLE_API_KEY / GEMINI_API_KEY instead of the given key, and from sending
// the key to a host named by GOOGLE_GEMINI_BASE_URL or SetDefaultBaseURLs.
// genai's HTTP retries are left off (its default), so a failed call fails
// once and the caller decides about retrying with llm.IsRetryable. genai's
// automatic stream continuation is left on: when Gemini ends a stream with
// finish reason CONTINUATION and a continuation token, the SDK resumes it
// and Generate sees one uninterrupted stream. A Provider is safe for
// concurrent use.
//
// # Requests
//
// Request.System becomes systemInstruction. Roles map user→"user" and
// assistant→"model". Blocks map to parts: text→text, image→inlineData,
// thinking→text with thought=true, tool_use→functionCall{id,name,args} and
// tool_result→functionResponse{id,name,response,parts}. Gemini needs the
// function's name in a functionResponse, so it is recovered from the
// tool_use with the same ID earlier in Request.Messages; a tool_result with
// no matching tool_use is an *llm.Error of kind invalid_request. The
// response object is {"output": text} or, when IsError, {"error": text},
// where text is the tool_result's text blocks joined with newlines. Images
// in a tool_result go into functionResponse.parts (multimodal function
// responses). Gemini 1.x/2.x models do not accept those, so for model IDs
// starting with "gemini-1" or "gemini-2" the images are sent as inlineData
// parts at the end of the same user turn instead. A message that maps to
// no parts (for example an empty assistant message after a blocked prompt)
// is skipped, and an empty unsigned text block is dropped, because Gemini
// rejects empty contents and empty parts. A block with an unknown Type, a
// tool_use whose Input is not a JSON object, a tool schema that is not
// valid JSON or corrupt ProviderRaw is an *llm.Error of kind
// invalid_request, returned before anything is sent.
//
// Every ToolSpec becomes a functionDeclaration in a single tool, with
// InputSchema sent unchanged as parametersJsonSchema. ToolSpec.Strict has no
// Gemini equivalent and is ignored. Request.CacheSystemAndTools is ignored
// too: Gemini caches repeated prompt prefixes implicitly and reports the
// hits as cachedContentTokenCount.
//
// Request.MaxTokens becomes maxOutputTokens; 0 uses WithDefaultMaxTokens if
// set, else Gemini's default. On Gemini, thinking tokens count against
// maxOutputTokens.
//
// # Effort
//
// thinkingConfig.includeThoughts is always true, so thought summaries
// stream as thinking blocks. For model IDs starting with "gemini-1" or
// "gemini-2" Effort sets thinkingBudget: low=1024, medium=8192 and
// high=24576 tokens (24576 is the largest budget every 2.5 model accepts:
// Flash and Flash-Lite cap at 24576, Pro at 32768; dynamic thinking (-1)
// is already the default, so it would make "high" mean nothing). Other
// effort values are an invalid_request error for these models. For all
// other models (Gemini 3 and later) Effort sets thinkingLevel to the
// upper-cased value: low→LOW, medium→MEDIUM, high→HIGH, and any other
// value is passed through (for example "minimal"→MINIMAL) for the API to
// accept or reject. An empty Effort leaves budget and level unset.
//
// # Responses and ProviderRaw
//
// Generate streams with streamGenerateContent. Consecutive streamed parts of
// the same kind (answer text or thought text) are merged into one text or
// thinking block, so Message.Text reads naturally. Each functionCall part
// becomes its own tool_use block. A functionCall without an ID (Gemini 2.x
// sends none) gets a generated ID "gemini_call_<random hex>".
//
// Thought signatures must go back to Gemini inside the part they arrived
// on, a signed part must never be merged with another part, and on Gemini 3
// function-call signatures are mandatory for multi-turn tool use. So a
// merged block remembers where the parts were. Within a block, consecutive
// unsigned parts form one segment and every signed part (often an
// empty-text part at the end of a stream) is a segment of its own. If any
// segment is signed, Block.ProviderRaw records them; Block.Text is always
// the full concatenation. ProviderRaw is this JSON object:
//
//	{
//	  "thought_signature": "<base64>", // tool_use: the functionCall part's signature
//	  "synthetic_id": true,            // tool_use: ID was generated here, not by Gemini
//	  "parts": [                       // text/thinking: segments, only when one is signed
//	    {"text": "...", "thought_signature": "<base64>"}
//	  ]
//	}
//
// Every field is omitted when empty, and blocks with nothing to record have
// no ProviderRaw. A thinking block whose segments carry a signature also
// gets Block.Signature = standard base64 of its first signature; that is
// informational, as the echo uses ProviderRaw.
//
// On the next request a text or thinking block with ProviderRaw.parts is
// sent as one part per segment, verbatim (thought=true for thinking),
// ignoring Block.Text. Without ProviderRaw it is one part holding
// Block.Text, and a thinking block's Block.Signature (standard base64), if
// set, becomes that part's thoughtSignature. A tool_use block's signature
// goes back on its functionCall part. A generated ID is never sent: the
// functionCall and the matching functionResponse go without an ID, as
// Gemini sent them.
//
// Response.Raw is not the raw stream. It is one synthesized
// GenerateContentResponse in Gemini's JSON shape: candidates[0] holds the
// assembled content exactly as it would be echoed, the finish reason and
// finish message; plus the last usageMetadata, modelVersion, responseId and
// promptFeedback. Model output parts other than text, thoughts and function
// calls (Gemini sends none unless server-side tools or image output are
// enabled, which this package never does) are dropped.
//
// Usage: InputTokens = promptTokenCount - cachedContentTokenCount +
// toolUsePromptTokenCount, CacheReadTokens = cachedContentTokenCount,
// OutputTokens = candidatesTokenCount + thoughtsTokenCount and
// CacheWriteTokens = 0 (implicit caching has no write charge).
//
// # Stop reasons
//
//	STOP, OTHER, unspecified, anything else  end_turn, or tool_use if the
//	                                         message has tool_use blocks
//	MAX_TOKENS, CONTINUATION                 max_tokens (CONTINUATION only
//	                                         reaches us if the SDK could
//	                                         not resume)
//	SAFETY, RECITATION, LANGUAGE, BLOCKLIST,
//	PROHIBITED_CONTENT, SPII, IMAGE_SAFETY,
//	IMAGE_PROHIBITED_CONTENT, IMAGE_RECITATION  refusal
//	promptFeedback.blockReason, no candidate refusal
//	MALFORMED_FUNCTION_CALL,
//	UNEXPECTED_TOOL_CALL                     *llm.Error kind server: the
//	                                         model produced a call that
//	                                         cannot be run; sending the
//	                                         same request again usually
//	                                         succeeds
//
// A stream that ends without a single chunk is an *llm.Error of kind
// server.
//
// # Errors
//
// API errors, whether returned as an HTTP status or sent as an error event
// inside the stream, become *llm.Error with Provider "gemini", the HTTP
// code as StatusCode, the API's status and message as Message (with the
// API key redacted, should it ever appear) and the genai.APIError as Err:
//
//	429 or RESOURCE_EXHAUSTED            rate_limited; RetryAfter from the
//	                                     google.rpc.RetryInfo detail's
//	                                     retryDelay, else the Retry-After
//	                                     header
//	503 or UNAVAILABLE                   overloaded
//	other 5xx, 408, INTERNAL, UNKNOWN,
//	DEADLINE_EXCEEDED, ABORTED, DATA_LOSS server
//	401, 403, UNAUTHENTICATED,
//	PERMISSION_DENIED, API_KEY_INVALID,
//	"API key not valid"/"expired"        auth
//	413, or a message saying the input
//	has too many tokens or is too large  context_too_long
//	other 4xx                            invalid_request
//
// Transport and stream-decoding failures are kind server. If ctx is done,
// Generate returns ctx.Err() itself.
package gemini
