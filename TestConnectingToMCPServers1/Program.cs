using Azure;
using Azure.AI.OpenAI;
using Microsoft.Extensions.AI;
using ModelContextProtocol.Client;
using ModelContextProtocol.Protocol.Transport;
using OpenAI;
using System.ClientModel;


//------------------
// Configuration
//------------------

string apiKey = ""; // Your OpenAI API key
string urlOfMcpServer = ""; // The URL of the MCP server you want to connect to

if (string.IsNullOrEmpty(apiKey))
{
    Console.WriteLine("Please set your OpenAI API key in the apiKey variable.");
    return;
}

if (string.IsNullOrEmpty(urlOfMcpServer))
{
    Console.WriteLine("Please set the URL of the MCP server in the urlOfMcpServer variable.");
    return;
}

var modelId = "gpt-4o-mini";

//------------------
// The actual code
//------------------

Console.WriteLine("Hello!");

var apiKeyCredential = new ApiKeyCredential(apiKey);

var openAIClient = new OpenAIClient(
    apiKeyCredential
);

IChatClient chatClient = openAIClient
    .GetChatClient(modelId)
    .AsIChatClient()
    .AsBuilder()
    .UseFunctionInvocation()
    .Build();

var transport = new SseClientTransport(new()
{
    Endpoint = new Uri(urlOfMcpServer),
});

// Create client and run tests
IMcpClient mcpClient = await McpClientFactory.CreateAsync(transport);
var mcpTools = await mcpClient.ListToolsAsync();
Console.ForegroundColor = ConsoleColor.DarkGray;
Console.WriteLine("Available tools:");
foreach (var tool in mcpTools)
{
    Console.WriteLine($"  {tool.Name}: {tool.Description}");
}

var chatHistory = new List<ChatMessage>();

var chatOptions = new ChatOptions
{
    Tools = [.. mcpTools]
};


// After creating the MCP client, let's inspect what methods are available
Console.WriteLine("=== MCP CLIENT INFO ===");
Console.WriteLine($"MCP Client Type: {mcpClient.GetType().FullName}");
Console.WriteLine($"MCP Client Interfaces: {string.Join(", ", mcpClient.GetType().GetInterfaces().Select(i => i.Name))}");

// List all methods to see what we have to work with
Console.WriteLine("\n=== MCP CLIENT METHODS ===");
foreach (var method in mcpClient.GetType().GetMethods())
{
    if (method.Name.Contains("Tool") || method.Name.Contains("Function"))
    {
        Console.WriteLine($"Method: {method.Name}");
        foreach (var parameter in method.GetParameters())
        {
            Console.WriteLine($"  Parameter: {parameter.Name} - Type: {parameter.ParameterType}");
        }
    }
}

Console.WriteLine("How can I help you today?");

while (true)
{
    var userPrompt = GetUserPrompt();
    if (string.IsNullOrEmpty(userPrompt)) break;

    chatHistory.Add(new ChatMessage(ChatRole.User, userPrompt));
    await GetResponse(chatClient, mcpClient, chatHistory, chatOptions);
    Console.WriteLine();
}

static string? GetUserPrompt()
{
    Console.ForegroundColor = ConsoleColor.DarkGray;
    Console.WriteLine("Your prompt:");
    Console.ForegroundColor = ConsoleColor.Yellow;
    var userPrompt = Console.ReadLine();
    return userPrompt;
}

static async Task<ChatMessage> GetResponse(IChatClient chatClient, IMcpClient mcpClient, List<ChatMessage> chatHistory, ChatOptions? options = null)
{
    Console.ForegroundColor = ConsoleColor.DarkGray;
    Console.WriteLine($"AI Response started at {DateTime.Now}:");

    // Debug - Print current chat history before processing
    Console.WriteLine("\n=== CHAT HISTORY BEFORE PROCESSING ===");
    for (int i = 0; i < chatHistory.Count; i++)
    {
        // Since ChatMessage doesn't have a Content property, use the standard ToString method
        Console.WriteLine($"[{i}] {chatHistory[i].Role}: {TruncateString(chatHistory[i].ToString(), 50)}");
    }

    string finalResponse = "";
    UsageDetails? usageDetails = null;

    // Get the entire response in one go (no loop)
    var allFunctionCalls = new List<FunctionCallContent>();
    string currentResponse = "";

    await foreach (var item in chatClient.GetStreamingResponseAsync(chatHistory, options))
    {
        // Collect any function calls we find
        var functionCallContent = item.Contents.FirstOrDefault(c => c is FunctionCallContent) as FunctionCallContent;
        if (functionCallContent != null)
        {
            allFunctionCalls.Add(functionCallContent);
            Console.ForegroundColor = ConsoleColor.Magenta;
            Console.WriteLine($"\nFunction call found: {functionCallContent.Name} (Collecting, not executing yet)");
        }
        else
        {
            // Regular text content
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.Write(item.Text);
            currentResponse += item.Text;
        }

        var usage = item.Contents.OfType<UsageContent>().FirstOrDefault()?.Details;
        if (usage != null) usageDetails = usage;
    }

    // Store the current assistant response even if we found function calls
    finalResponse = currentResponse;

    // If we found function calls, process them and return their responses only, without another AI call
    if (allFunctionCalls.Count > 0)
    {
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine($"\nProcessing {allFunctionCalls.Count} function call(s) without adding to chat history yet");

        // Process each function call sequentially
        foreach (var functionCallContent in allFunctionCalls)
        {
            Console.WriteLine($"\nExecuting MCP tool: {functionCallContent.Name}");

            try
            {
                // Get the arguments from the function call
                var arguments = functionCallContent.Arguments as Dictionary<string, object>;

                // Display the arguments
                if (arguments != null)
                {
                    Console.WriteLine("Arguments:");
                    foreach (var arg in arguments)
                    {
                        Console.WriteLine($"  {arg.Key}: {arg.Value}");
                    }
                }

                // Call the MCP tool
                var toolResult = await mcpClient.CallToolAsync(
                    functionCallContent.Name,
                    arguments
                );

                // Extract the text content from the result
                var toolResponseText = string.Join("\n",
                    toolResult.Content
                        .Where(c => c.Type == "text")
                        .Select(c => c.Text));

                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine($"Tool response: {toolResponseText}");

                // Do NOT add to chat history yet
                Console.WriteLine($"Tool response from {functionCallContent.Name} will be returned to the user directly");

                // Append tool response to final response
                finalResponse += $"\n\n**Results from {functionCallContent.Name}:**\n{toolResponseText}";
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"Error executing tool {functionCallContent.Name}: {ex.Message}");

                // Append error to final response
                finalResponse += $"\n\n**Error from {functionCallContent.Name}:**\n{ex.Message}";
            }
        }
    }

    // Add only one final assistant response that includes both AI text and tool results
    var assistantMessage = new ChatMessage(ChatRole.Assistant, finalResponse);
    chatHistory.Add(assistantMessage);

    Console.ForegroundColor = ConsoleColor.DarkGray;
    Console.WriteLine($"\nAI Response completed at {DateTime.Now}:");
    ShowUsageDetails(usageDetails);

    // Debug - Print final chat history after processing
    Console.WriteLine("\n=== CHAT HISTORY AFTER PROCESSING ===");
    for (int i = 0; i < chatHistory.Count; i++)
    {
        // Since ChatMessage doesn't have a Content property, use the standard ToString method
        Console.WriteLine($"[{i}] {chatHistory[i].Role}: {TruncateString(chatHistory[i].ToString(), 50)}");
    }

    return assistantMessage;
}

// Helper method to truncate strings
static string TruncateString(string input, int maxLength)
{
    if (string.IsNullOrEmpty(input)) return string.Empty;
    return input.Length <= maxLength ? input : input.Substring(0, maxLength) + "...";
}


static void ShowUsageDetails(UsageDetails? usage)
{
    if (usage != null)
    {
        Console.WriteLine($"  InputTokenCount: {usage.InputTokenCount}");
        Console.WriteLine($"  OutputTokenCount: {usage.OutputTokenCount}");
        Console.WriteLine($"  TotalTokenCount: {usage.TotalTokenCount}");

        if (usage.AdditionalCounts != null)
        {
            foreach (var additionalCount in usage.AdditionalCounts)
            {
                Console.WriteLine($"  {additionalCount.Key}: {additionalCount.Value}");
            }

        }
    }
}
