import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime as dt
import logging
from typing import Literal, Optional

import discord
import httpx
from openai import AsyncOpenAI
import yaml
from retriever import retriver

from llama_index.core import VectorStoreIndex, StorageContext, ServiceContext, PromptTemplate, get_response_synthesizer
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.chat_engine.types import ChatMode
from llama_index.vector_stores.qdrant import QdrantVectorStore

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.groq import Groq
from llama_index.llms.ollama import Ollama
from llama_index.llms.openrouter import OpenRouter


from fastembed import SparseTextEmbedding
import qdrant_client
from qdrant_client import QdrantClient, models

import asyncstdlib as a

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

VISION_MODEL_TAGS = ("gpt-4o", "claude-3", "gemini", "pixtral", "llava", "vision", "vl")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

ALLOWED_FILE_TYPES = ("image", "text")
ALLOWED_CHANNEL_TYPES = (discord.ChannelType.text, discord.ChannelType.public_thread, discord.ChannelType.private_thread, discord.ChannelType.private)

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

MAX_MESSAGE_NODES = 100


def get_config(filename="config.yaml"):
    with open(filename, "r") as file:
        return yaml.safe_load(file)


cfg = get_config()

if client_id := cfg["client_id"]:
    logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/api/oauth2/authorize?client_id={client_id}&permissions=412317273088&scope=bot\n")

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(cfg["status_message"] or "github.com/jakobdylanc/llmcord")[:128])
discord_client = discord.Client(intents=intents, activity=activity)

httpx_client = httpx.AsyncClient()

search = retriver(collection_name="Digital_IC_A_D", score_threshold=0.5)

msg_nodes = {}
last_task_time = None

qa_prompt_tmpl = (
    "Context information is below.\n"
    "-------------------------------"
    "{context_str}\n"
    "-------------------------------"
    "Given the context information and not prior knowledge,"
    "answer the query. Please be concise, and complete.\n"
    "If the context does not contain an answer to the query,"
    "respond with \"Sorry, I don't know!\"."
    "Query: {query_str}\n"
    "Answer: "
)
qa_prompt = PromptTemplate(qa_prompt_tmpl)

def create_sparse_vector(query_list):
    """
        Create a sparse vector from the text using BM25.
    """
    model = SparseTextEmbedding(model_name="Qdrant/bm25")

    indices_list = []
    values_list  = []
    for text in query_list:
        embeddings = list(model.embed(text))[0]

        #sparse_vector = models.SparseVector(
        indices=embeddings.indices.tolist()
        values=embeddings.values.tolist()

        indices_list.append(indices)
        values_list.append(values)

    return indices_list, values_list

ollama_embedding = OllamaEmbedding(
    model_name="bge-m3:latest",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

client = qdrant_client.QdrantClient(
    host="localhost",
    port=6333
)

aclient = qdrant_client.AsyncQdrantClient(
    host="localhost",
    port=6333
)

vector_store = QdrantVectorStore(client=client,
                                 aclient=aclient, 
                                 #collection_name="cmos_vlsi_4ed",
                                 collection_name="Digital_IC_A_D",
                                 #collection_name="collection_bm25_256_0",
                                 #fastembed_sparse_model="Qdrant/bm25",
                                 sparse_doc_fn=create_sparse_vector,
                                 sparse_query_fn=create_sparse_vector,
                                 #hybrid_fusion_fn="rrf",
                                 enable_hybrid=True)
#try:
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(
    embed_model=ollama_embedding,
    vector_store=vector_store,
    use_async=True
)

memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

provider, model = cfg["model"].split("/", 1)
base_url = cfg["providers"][provider]["base_url"]
api_key = cfg["providers"][provider].get("api_key", "sk-no-key-required")
if provider == 'groq':
    llm = Groq(model=model, api_key=api_key)
elif provider == 'openrouter':
    llm = OpenRouter(model=model, api_key=api_key)
elif provider == 'ollama':
    llm = Ollama(model=model, base_url=base_url)
else:
    raise ValueError("Provider not supported.")

#response_synthesizer = get_response_synthesizer(
#    llm=GEN_MODEL,
#    text_qa_template=qa_prompt,
#    response_mode="compact",
#)

#from llama_index.core import Settings
#
#from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler, CBEventType
#
#llama_debug = LlamaDebugHandler(print_trace_on_end=True)
#callback_manager = CallbackManager([llama_debug])
#Settings.callback_manager=callback_manager

chat_engine = index.as_chat_engine(
    #response_synthesizer=response_synthesizer,
    memory=memory,
    chat_mode=ChatMode.CONDENSE_PLUS_CONTEXT, 
    #chat_mode=ChatMode.SIMPLE, 
    #chat_mode=ChatMode.CONDENSE_QUESTION,
    llm=llm, 
    verbose=True,
    streaming=True,
)



@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list = field(default_factory=list)

    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None

    next_msg: Optional[discord.Message] = None

    has_bad_attachments: bool = False
    fetch_next_failed: bool = False

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@discord_client.event
async def on_message(new_msg):
    global msg_nodes, last_task_time

    if (
        new_msg.channel.type not in ALLOWED_CHANNEL_TYPES
        or (new_msg.channel.type != discord.ChannelType.private and discord_client.user not in new_msg.mentions)
        or new_msg.author.bot
    ):
        return

    cfg = get_config()
    
    allowed_channel_ids = cfg["allowed_channel_ids"]
    allowed_role_ids = cfg["allowed_role_ids"]

    if (allowed_channel_ids and not any(id in allowed_channel_ids for id in (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None)))) or (
        allowed_role_ids and not any(role.id in allowed_role_ids for role in getattr(new_msg.author, "roles", []))
    ):
        return
    
    accept_images: bool = any(x in model.lower() for x in VISION_MODEL_TAGS)
    accept_usernames: bool = any(x in provider.lower() for x in PROVIDERS_SUPPORTING_USERNAMES)

    max_text = cfg["max_text"]
    max_images = cfg["max_images"] if accept_images else 0
    max_messages = cfg["max_messages"]

    use_plain_responses: bool = cfg["use_plain_responses"]
    max_message_length = 2000 if use_plain_responses else (4096 - len(STREAMING_INDICATOR))

    # Build message chain and set user warnings
    messages = []
    user_warnings = set()
    curr_msg = new_msg
    while curr_msg != None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text == None:
                good_attachments = {type: [att for att in curr_msg.attachments if att.content_type and type in att.content_type] for type in ALLOWED_FILE_TYPES}

                curr_node.text = "\n".join(
                    ([curr_msg.content] if curr_msg.content else [])
                    + [embed.description for embed in curr_msg.embeds if embed.description]
                    + [(await httpx_client.get(att.url)).text for att in good_attachments["text"]]
                )
                if curr_node.text.startswith(discord_client.user.mention):
                    curr_node.text = curr_node.text.replace(discord_client.user.mention, "", 1).lstrip()

                curr_node.images = [
                    dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode((await httpx_client.get(att.url)).content).decode('utf-8')}"))
                    for att in good_attachments["image"]
                ]

                curr_node.role = "assistant" if curr_msg.author == discord_client.user else "user"

                curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None

                curr_node.has_bad_attachments = len(curr_msg.attachments) > sum(len(att_list) for att_list in good_attachments.values())

                try:
                    if (
                        curr_msg.reference == None
                        and discord_client.user.mention not in curr_msg.content
                        and (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                        and any(prev_msg_in_channel.type == type for type in (discord.MessageType.default, discord.MessageType.reply))
                        and prev_msg_in_channel.author == (discord_client.user if curr_msg.channel.type == discord.ChannelType.private else curr_msg.author)
                    ):
                        curr_node.next_msg = prev_msg_in_channel
                    else:
                        next_is_thread_parent: bool = curr_msg.reference == None and curr_msg.channel.type == discord.ChannelType.public_thread
                        if next_msg_id := curr_msg.channel.id if next_is_thread_parent else getattr(curr_msg.reference, "message_id", None):
                            if next_is_thread_parent:
                                curr_node.next_msg = curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(next_msg_id)
                            else:
                                curr_node.next_msg = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(next_msg_id)

                except (discord.NotFound, discord.HTTPException, AttributeError):
                    logging.exception("Error fetching next message in the chain")
                    curr_node.fetch_next_failed = True

            if curr_node.images[:max_images]:
                content = ([dict(type="text", text=curr_node.text[:max_text])] if curr_node.text[:max_text] else []) + curr_node.images[:max_images]
            else:
                content = curr_node.text[:max_text]

            if content != "":
                message = dict(content=content, role=curr_node.role)
                if accept_usernames and curr_node.user_id != None:
                    message["name"] = str(curr_node.user_id)

                messages.append(message)

            if len(curr_node.text) > max_text:
                user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if len(curr_node.images) > max_images:
                user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
            if curr_node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_next_failed or (curr_node.next_msg != None and len(messages) == max_messages):
                user_warnings.add(f"⚠️ Only using last {len(messages)} message{'' if len(messages) == 1 else 's'}")

            curr_msg = curr_node.next_msg

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")

    #print(len(messages))
    #print(messages)
    chat_history=[]
    for _, message in enumerate(messages[:0:-1]):
    #for _, message in enumerate(messages[::-1]):
        chat_history.append(ChatMessage.from_str(role=message['role'], content=message['content']))
    #print(chat_history)
    #result = chat_engine.chat(
    #             message=messages[0]["content"],
    #              chat_history=chat_history
    #            )
    #print(f"[DBG]{result}\n")
    
    response = await chat_engine.astream_chat(
                message=messages[0]["content"],
                chat_history=chat_history
            )
    
    # Generate and send response message(s) (can be multiple if response is long)
    response_msgs = []
    response_contents = []
    prev_chunk = None
    edit_task = None
    just_sent = False 
    try:
        async with new_msg.channel.typing():
            #response = await qa
            #responses = result.async_response_gen()
            async for idx, curr_chunk in a.enumerate(response.async_response_gen()):
            #async for idx, curr_chunk in a.enumerate(response.achat_stream):
                print(curr_chunk)
                #if response.is_function_false_event is not None:
                #    print(f"[IDX]{idx} {response_contents} {response.is_done}")
                #continue    
            #async for curr_chunk in enumerate(result):
            #async for curr_chunk in await qa.response:
            #async for curr_chunk in await qa.response:
                #prev_content = prev_chunk.choices[0].delta.content if prev_chunk != None and prev_chunk.choices[0].delta.content else ""
                #curr_content = curr_chunk.choices[0].delta.content or ""
                prev_content = prev_chunk if prev_chunk != None and prev_chunk else ""
                curr_content = curr_chunk or ""
                #event_pairs = llama_debug.get_llm_inputs_outputs()

                if response_contents or prev_content:
                    if response_contents == [] or len(response_contents[-1] + prev_content) > max_message_length:
                        response_contents.append("")
                        
                        #print(f"{len(response_contents[-1] + prev_content)}")
                        if not use_plain_responses:
                            #print(f"[PREV][{prev_content}]")
                            embed = discord.Embed(description=(prev_content + STREAMING_INDICATOR), color=EMBED_COLOR_INCOMPLETE)
                            for warning in sorted(user_warnings):
                                embed.add_field(name=warning, value="", inline=False)

                            reply_to_msg = new_msg if response_msgs == [] else response_msgs[-1]
                            response_msg = await reply_to_msg.reply(embed=embed, silent=True)
                            msg_nodes[response_msg.id] = MsgNode(next_msg=new_msg)
                            await msg_nodes[response_msg.id].lock.acquire()
                            response_msgs.append(response_msg)
                            last_task_time = dt.now().timestamp()

                    response_contents[-1] += prev_content

                    if not use_plain_responses:
                        #finish_reason = curr_chunk.choices[0].finish_reason
                        finish_reason = None
                        #try:
                        #if len(event_pairs[1])==2:
                        #    finish_reason = event_pairs[1][1].payload['response'].raw.choices[0].finish_reason
                        #    print(f"[EVENT 11]{finish_reason}")
                        #else:
                        #    finish_reason = None
                        #except:
                        #    finish_reason = None
                        #try:
                        #print(event_pairs[1][0].payload['completion'].raw.choices[0].finish_reason)
                        #except:
                        #    print(event_pairs[0][1].payload['completion'])

                        ready_to_edit: bool = (edit_task == None or edit_task.done()) and dt.now().timestamp() - last_task_time >= EDIT_DELAY_SECONDS
                        msg_split_incoming: bool = len(response_contents[-1] + curr_content) > max_message_length
                        is_final_edit: bool = finish_reason != None or msg_split_incoming
                        is_good_finish: bool = (finish_reason != None and any(finish_reason.lower() == x for x in ("stop", "end_turn")))
                        #is_final_edit: bool = response.is_done or msg_split_incoming
                        #is_good_finish: bool = response.is_done

                        if ready_to_edit or is_final_edit:
                            if edit_task != None:
                                await edit_task
                            print(f"[DONE] [{response_msgs[-1]}]")
                            embed.description = response_contents[-1] if is_final_edit else (response_contents[-1] + STREAMING_INDICATOR)
                            embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE
                            edit_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))
                            last_task_time = dt.now().timestamp()
                            just_sent = True
                        else:
                            just_sent = False

                #for i in range(len(event_pairs)):
                #    print(f"[{i}] {len(event_pairs[i])}")
                
                #print(f"[REASON]{str(event_pairs[1][0].payload['completion'].raw.choices[0].finish_reason)}")
                #print(f"[EVENT 00]{event_pairs[0][0]}")
                #print(f"[EVENT 01]{event_pairs[0][1]}")
                #print(f"[EVENT 10]{event_pairs[1][0]}")
                prev_chunk = curr_chunk
        
        #if response.is_function_false_event is not None:
        #    print(f"[FIN] {response_contents} {response.is_done}")
        

        if use_plain_responses :
            for content in response_contents:
                reply_to_msg = new_msg if response_msgs == [] else response_msgs[-1]
                response_msg = await reply_to_msg.reply(content=content, suppress_embeds=True)
                msg_nodes[response_msg.id] = MsgNode(next_msg=new_msg)
                await msg_nodes[response_msg.id].lock.acquire()
                response_msgs.append(response_msg)
        else:
            if edit_task != None:
                await edit_task
            print(f"[LAST] [{response_msgs[-1]}]")
            embed.description = response_contents[-1] if just_sent == False else ""
            embed.color = EMBED_COLOR_COMPLETE
            edit_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))
            last_task_time = dt.now().timestamp()

        if edit_task != None:
            await edit_task

    except:
        logging.exception("Error while generating response")

    print(response_msgs)
    for msg in response_msgs:
        msg_nodes[msg.id].text = "".join(response_contents)
        msg_nodes[msg.id].lock.release()

    # Delete oldest MsgNodes (lowest message IDs) from the cache
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)


async def main():
    await discord_client.start(cfg["bot_token"])


asyncio.run(main())
