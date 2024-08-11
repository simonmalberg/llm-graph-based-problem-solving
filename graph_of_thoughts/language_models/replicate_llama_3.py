# Copyright (c) 2023 ETH Zurich.
# All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Nils Blach

import backoff
import os
from typing import List, Dict, Union
from replicate import Client

from .abstract_language_model import AbstractLanguageModel

class ReplicateError(Exception):
    pass

class ReplicateLanguageModel(AbstractLanguageModel):
    """
    The ReplicateLanguageModel class handles interactions with the Replicate API using the provided configuration.

    Inherits from AbstractLanguageModel and implements its abstract methods.
    """

    def __init__(
        self, config_path: str = "", model_name: str = "meta-llama-3-8b-instruct", cache: bool = False
    ) -> None:
        """
        Initialize the ReplicateLanguageModel instance with configuration, model details, and caching options.

        :param config_path: Path to the configuration file. Defaults to "".
        :type config_path: str
        :param model_name: Name of the model, default is 'meta-llama-3-8b-instruct'. Used to select the correct configuration.
        :type model_name: str
        :param cache: Flag to determine whether to cache responses. Defaults to False.
        :type cache: bool
        """
        super().__init__(config_path, model_name, cache)
        self.config: Dict = self.config[model_name]
        self.model_id: str = self.config["model_id"]
        self.prompt_token_cost: float = self.config["prompt_token_cost"]
        self.response_token_cost: float = self.config["response_token_cost"]
        self.temperature: float = self.config["temperature"]
        self.max_tokens: int = self.config["max_tokens"]
        self.top_k: int = self.config["top_k"]
        self.stop: Union[str, List[str]] = self.config["stop"]
        self.prompt_template: str = self.config.get("prompt_template", "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
        self.api_key: str = os.getenv("REPLICATE_API_TOKEN", self.config["api_key"])
        if not self.api_key:
            raise ValueError("REPLICATE_API_TOKEN is not set")
        self.client = Client(api_token=self.api_key)

    def query(
        self, query: str, num_responses: int = 1
    ) -> Union[List[str], str]:
        """
        Query the Replicate model for responses.

        :param query: The query to be posed to the language model.
        :type query: str
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: Response(s) from the Replicate model.
        :rtype: Union[List[str], str]
        """
        if self.cache and query in self.respone_cache:
            return self.respone_cache[query]

        try:
            response = [self.chat(query) for _ in range(num_responses)]
        except ReplicateError as e:
            self.logger.warning(f"Error in Replicate API: {e}, retrying...")
            response = []

        if self.cache:
            self.respone_cache[query] = response
        return response[0] if num_responses == 1 else response

    @backoff.on_exception(backoff.expo, ReplicateError, max_time=600, max_tries=60)
    def chat(self, prompt: str) -> str:
        """
        Send chat messages to the Replicate model and retrieve the model's response.
        Implements backoff on Replicate error.

        :param prompt: The prompt to be sent to the model.
        :type prompt: str
        :return: The model's response.
        :rtype: str
        """
        formatted_prompt = self.prompt_template.format(prompt=prompt)
        input_data = {
            "prompt": formatted_prompt,
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "stop": self.stop,
        }

        try:
            prediction = self.client.predictions.create(model=self.model_id, input=input_data)
            prediction.wait()
            if prediction.status == "failed":
                raise ReplicateError(f"Error in Replicate API. Prediction: {prediction}")
        except Exception as e:
            raise ReplicateError(f"Error in Replicate API: {e}")
        self.prompt_tokens += prediction.metrics["input_token_count"]
        self.completion_tokens += prediction.metrics["output_token_count"]
        prompt_tokens_k = float(self.prompt_tokens) / 1000.0
        completion_tokens_k = float(self.completion_tokens) / 1000.0
        self.cost = (
            self.prompt_token_cost * prompt_tokens_k
            + self.response_token_cost * completion_tokens_k
        )
        return "".join(prediction.output)

    def get_response_texts(
        self, query_response: Union[List[str], str]
    ) -> List[str]:
        """
        Extract the response texts from the query response.

        :param query_response: The response(s) from the Replicate model.
        :type query_response: Union[List[str], str]
        :return: List of response strings.
        :rtype: List[str]
        """
        if isinstance(query_response, str):
            return [query_response]
        return query_response
