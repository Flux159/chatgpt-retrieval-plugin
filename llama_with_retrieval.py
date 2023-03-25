from typing import List
import requests
import openai
import os

from tenacity import retry, wait_random_exponential, stop_after_attempt


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def get_context(prompt: str) -> List[str]:
    """
    Queries the data store using the retrieval plugin to get relevant context.

    Args:
        prompt: The user prompt to identify context for.

    Returns:
        A list of document chunks from the data store, sorted by proximity of vector similarity.
    """

    retrieval_endpoint = os.environ.get("DATASTORE_QUERY_URL")
    bearer_token = os.environ.get("BEARER_TOKEN")

    # curl -X 'POST' \
    #   'http://0.0.0.0:8000/query' \
    #   -H 'accept: application/json' \
    #   -H 'Authorization: Bearer test1234' \
    #   -H 'Content-Type: application/json' \
    #   -d '{
    #   "queries": [
    #     {
    #       "query": "How do I activate Conda?",
    #       "filter": {
    #         "document_id": "4827d5ac-2875-40ac-9279-dab0964cbf5a"
    #       },
    #       "top_k": 3
    #     }
    #   ]
    # }'


    headers = {
        "Content-Type": "application/json", 
        "Authorization": f"Bearer {bearer_token}"
    }

    data = {
        "queries": [
            {
                "query": prompt,
                #"filter": { "document_id": "4827d5ac-2875-40ac-9279-dab0964cbf5a"},
                "top_k": 3
            }
        ]
    }

    print(f"url={retrieval_endpoint}")

    response = requests.post(url=retrieval_endpoint, data=data, headers=headers)

    print(response.text)


    # Call the OpenAI API to get the embeddings
    # response = openai.Embedding.create(input=texts, model="text-embedding-ada-002")

    # # Extract the embedding data from the response
    # data = response["data"]  # type: ignore

    # # Return the embeddings as a list of lists of floats
    # return [result["embedding"] for result in data]


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def get_chat_completion(
    messages,
    model="gpt-3.5-turbo",  # use "gpt-4" for better results
):
    """
    Generate a chat completion using OpenAI's chat completion API.

    Args:
        messages: The list of messages in the chat history.
        model: The name of the model to use for the completion. Default is gpt-3.5-turbo, which is a fast, cheap and versatile model. Use gpt-4 for higher quality but slower results.

    Returns:
        A string containing the chat completion.

    Raises:
        Exception: If the OpenAI API call fails.
    """
    # call the OpenAI chat completion API with the given messages
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
    )

    choices = response["choices"]  # type: ignore
    completion = choices[0].message.content.strip()
    print(f"Completion: {completion}")
    return completion


get_context("How do I activate Conda for my project?")