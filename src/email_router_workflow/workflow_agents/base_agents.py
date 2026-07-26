import ast
from pathlib import Path

import numpy as np
import pandas as pd
import re
import csv
import uuid
from datetime import datetime

from ..openai_config import create_openai_client


class DirectPromptAgent:
    """
    A minimal agent that sends a user prompt directly to an LLM without
    additional context, persona, or system instructions.
    """

    def __init__(self, openai_api_key):
        """
        Initialize the DirectPromptAgent.

        Parameters:
        openai_api_key (str): API key used to authenticate OpenAI requests.
        """
        self.openai_api_key = openai_api_key

    def respond(self, prompt):
        """
        Send a user prompt to the LLM and return the generated text response.

        Parameters:
        prompt (str): The user's input prompt.

        Returns:
        str: Text content from the LLM response.
        """
        client = create_openai_client(self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content


class AugmentedPromptAgent:
    """
    An agent that applies a predefined persona through a system prompt while
    responding to user input.
    """

    def __init__(self, openai_api_key, persona):
        """
        Initialize the AugmentedPromptAgent.

        Parameters:
        openai_api_key (str): API key used to authenticate OpenAI requests.
        persona (str): Persona description applied through the system prompt.
        """
        self.persona = persona
        self.openai_api_key = openai_api_key

    def respond(self, input_text):
        """
        Generate a response using the configured persona.

        Parameters:
        input_text (str): The user's input prompt.

        Returns:
        str: Text content from the LLM response.
        """
        client = create_openai_client(self.openai_api_key)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are {self.persona}. Forget all previous context."},
                {"role": "user", "content": input_text}
            ],
            temperature=0
        )

        return response.choices[0].message.content


class KnowledgeAugmentedPromptAgent:
    """
    An agent that combines a persona with explicit provided knowledge and
    instructs the LLM to answer using only that knowledge.
    """

    def __init__(self, openai_api_key, persona, knowledge):
        """
        Initialize the KnowledgeAugmentedPromptAgent.

        Parameters:
        openai_api_key (str): API key used to authenticate OpenAI requests.
        persona (str): Persona description applied through the system prompt.
        knowledge (str): Domain knowledge the agent must use when answering.
        """
        self.persona = persona
        self.knowledge = knowledge
        self.openai_api_key = openai_api_key

    def respond(self, input_text):
        """
        Generate a response grounded in the agent's provided knowledge.

        Parameters:
        input_text (str): The user's input prompt.

        Returns:
        str: Text content from the LLM response.
        """
        client = create_openai_client(self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are {self.persona} knowledge-based assistant. Forget all previous context. "
                        f"Use only the following knowledge to answer, do not use your own knowledge: {self.knowledge} "
                        "Answer the prompt based on this knowledge, not your own."
                    )
                },
                {"role": "user", "content": input_text}
            ],
            temperature=0
        )
        return response.choices[0].message.content


class RAGKnowledgePromptAgent:
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(self, openai_api_key, persona, chunk_size=2000, chunk_overlap=100):
        """
        Initializes the RAGKnowledgePromptAgent with API credentials and configuration settings.

        Parameters:
        openai_api_key (str): API key for accessing OpenAI.
        persona (str): Persona description for the agent.
        chunk_size (int): The size of text chunks for embedding. Defaults to 2000.
        chunk_overlap (int): Overlap between consecutive chunks. Defaults to 100.
        """
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"
        self.output_dir = Path.cwd() / "artifacts" / "rag"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_embedding(self, text):
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.

        Parameters:
        text (str): Text to embed.

        Returns:
        list: The embedding vector.
        """
        client = create_openai_client(self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        """
        Calculates cosine similarity between two vectors.

        Parameters:
        vector_one (list): First embedding vector.
        vector_two (list): Second embedding vector.

        Returns:
        float: Cosine similarity between vectors.
        """
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def chunk_text(self, text):
        """
        Splits text into manageable chunks, attempting natural breaks.

        Parameters:
        text (str): Text to split into chunks.

        Returns:
        list: List of dictionaries containing chunk metadata.
        """
        separator = "\n"
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) <= self.chunk_size:
            return [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]

        chunks, start, chunk_id = [], 0, 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if separator in text[start:end]:
                end = start + text[start:end].rindex(separator) + len(separator)

            chunks.append({
                "chunk_id": chunk_id,
                "text": text[start:end],
                "chunk_size": end - start,
                "start_char": start,
                "end_char": end
            })

            if end == len(text):
                break

            start = end - self.chunk_overlap
            chunk_id += 1

        with open(self.output_dir / f"chunks-{self.unique_filename}", 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for chunk in chunks:
                writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})

        return chunks

    def calculate_embeddings(self):
        """
        Calculates embeddings for each chunk and stores them in a CSV file.

        Returns:
        DataFrame: DataFrame containing text chunks and their embeddings.
        """
        chunks_path = self.output_dir / f"chunks-{self.unique_filename}"
        df = pd.read_csv(chunks_path, encoding='utf-8')
        df['embeddings'] = df['text'].apply(self.get_embedding)
        embeddings_path = self.output_dir / f"embeddings-{self.unique_filename}"
        df.to_csv(embeddings_path, encoding='utf-8', index=False)
        return df

    def find_prompt_in_knowledge(self, prompt):
        """
        Finds and responds to a prompt based on similarity with embedded knowledge.

        Parameters:
        prompt (str): User input prompt.

        Returns:
        str: Response derived from the most similar chunk in knowledge.
        """
        prompt_embedding = self.get_embedding(prompt)
        embeddings_path = self.output_dir / f"embeddings-{self.unique_filename}"
        df = pd.read_csv(embeddings_path, encoding='utf-8')
        df['embeddings'] = df['embeddings'].apply(lambda x: np.array(ast.literal_eval(x)))
        df['similarity'] = df['embeddings'].apply(lambda emb: self.calculate_similarity(prompt_embedding, emb))

        best_chunk = df.loc[df['similarity'].idxmax(), 'text']

        client = create_openai_client(self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are {self.persona}, a knowledge-based assistant. Forget previous context."},
                {"role": "user", "content": f"Answer based only on this information: {best_chunk}. Prompt: {prompt}"}
            ],
            temperature=0
        )

        return response.choices[0].message.content


class EvaluationAgent:
    """
    Iteratively evaluates worker agent responses against defined criteria and
    generates correction instructions until the response passes or max_interactions
    is reached.
    """

    def __init__(self, openai_api_key, persona, evaluation_criteria, worker_agent, max_interactions):
        """
        Initialize the EvaluationAgent.

        Parameters:
        openai_api_key (str): API key used to authenticate OpenAI requests.
        persona (str): Persona used by the evaluator when judging responses.
        evaluation_criteria (str): Criteria the worker response must satisfy.
        worker_agent: Agent whose responses will be evaluated and refined.
        max_interactions (int): Maximum number of evaluation/refinement loops.
        """
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions

    def evaluate(self, initial_prompt, worker_response=None):
        """
        Run the evaluator-optimizer loop for a worker agent response.

        Parameters:
        initial_prompt (str): Original prompt given to the worker agent.
        worker_response (str, optional): Pre-generated worker response to evaluate first.

        Returns:
        dict: Dictionary containing final_response, evaluation, and iterations.
        """
        client = create_openai_client(self.openai_api_key)
        prompt_to_evaluate = initial_prompt
        final_response = None
        final_evaluation = None
        iterations = 0
        pending_worker_response = worker_response

        for i in range(self.max_interactions):
            print(f"\n--- Interaction {i+1} ---")

            print(" Step 1: Worker agent generates a response to the prompt")
            print(f"Prompt:\n{prompt_to_evaluate}")
            if pending_worker_response is not None:
                response_from_worker = pending_worker_response
                pending_worker_response = None
            else:
                response_from_worker = self.worker_agent.respond(prompt_to_evaluate)
            print(f"Worker Agent Response:\n{response_from_worker}")

            print(" Step 2: Evaluator agent judges the response")
            eval_prompt = (
                f"Does the following answer: {response_from_worker}\n"
                f"Meet this criteria: {self.evaluation_criteria}\n"
                "Respond Yes ONLY if the answer meets every requirement in the criteria. "
                "Respond No if any requirement is missing or the answer uses the wrong deliverable format. "
                "Respond No if the answer uses numbered lists, bullet points, or section headings "
                "when the criteria require labeled fields or user story sentences. "
                "Respond No if the answer contains instructions instead of actual deliverables. "
                "Respond Yes or No, and the reason why it does or doesn't meet the criteria."
            )
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.persona},
                    {"role": "user", "content": eval_prompt}
                ],
                temperature=0
            )
            evaluation = response.choices[0].message.content.strip()
            print(f"Evaluator Agent Evaluation:\n{evaluation}")

            final_response = response_from_worker
            final_evaluation = evaluation
            iterations = i + 1

            print(" Step 3: Check if evaluation is positive")
            if evaluation.lower().startswith("yes"):
                print("Final solution accepted.")
                break
            else:
                print(" Step 4: Generate instructions to correct the response")
                instruction_prompt = (
                    f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation}"
                )
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": self.persona},
                        {"role": "user", "content": instruction_prompt}
                    ],
                    temperature=0
                )
                instructions = response.choices[0].message.content.strip()
                print(f"Instructions to fix:\n{instructions}")

                print(" Step 5: Send feedback to worker agent for refinement")
                prompt_to_evaluate = (
                    f"The original prompt was: {initial_prompt}\n"
                    f"The response to that prompt was: {response_from_worker}\n"
                    f"It has been evaluated as incorrect.\n"
                    f"Make only these corrections, do not alter content validity: {instructions}\n"
                    f"Rewrite the full answer so it meets every requirement in the evaluation criteria. "
                    f"Do not use numbered lists, bullet points, or section headings unless the criteria "
                    f"require specific field labels."
                )

        return {
            "final_response": final_response,
            "evaluation": final_evaluation,
            "iterations": iterations
        }


class RoutingAgent:
    """
    Routes user prompts to the most appropriate specialized agent based on
    semantic similarity between the prompt and agent descriptions.
    """

    def __init__(self, openai_api_key, agents):
        """
        Initialize the RoutingAgent.

        Parameters:
        openai_api_key (str): API key used to authenticate OpenAI requests.
        agents (list): Route definitions containing name, description, and func.
        """
        self.openai_api_key = openai_api_key
        self.agents = agents

    def get_embedding(self, text):
        """
        Calculate an embedding vector for the provided text.

        Parameters:
        text (str): Text to embed.

        Returns:
        list: Embedding vector returned by the OpenAI embeddings API.
        """
        client = create_openai_client(self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        embedding = response.data[0].embedding
        return embedding

    def route(self, user_input):
        """
        Select the best agent for a prompt and return that agent's response.

        Parameters:
        user_input (str): Prompt or workflow step to route.

        Returns:
        str: Response produced by the selected agent's callable function.
        """
        input_emb = self.get_embedding(user_input)
        best_agent = None
        best_score = -1

        for agent in self.agents:
            agent_emb = self.get_embedding(agent["description"])
            if agent_emb is None:
                continue

            similarity = np.dot(input_emb, agent_emb) / (np.linalg.norm(input_emb) * np.linalg.norm(agent_emb))
            print(similarity)

            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")
        return best_agent["func"](user_input)


class ActionPlanningAgent:
    """
    Extracts actionable workflow steps from a user prompt using predefined
    planning knowledge and an LLM system prompt.
    """

    def __init__(self, openai_api_key, knowledge):
        """
        Initialize the ActionPlanningAgent.

        Parameters:
        openai_api_key (str): API key used to authenticate OpenAI requests.
        knowledge (str): Planning knowledge used to derive workflow steps.
        """
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge

    def extract_steps_from_prompt(self, prompt):
        """
        Extract and return a cleaned list of action steps from a user prompt.

        Parameters:
        prompt (str): High-level workflow request from the user.

        Returns:
        list: Non-empty step strings derived from the LLM response.
        """
        client = create_openai_client(self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an action planning agent. Using your knowledge, you extract from the user prompt "
                        "the steps requested to complete the action the user is asking for. You return the steps as a list. "
                        f"Only return the steps in your knowledge. Forget any previous context. This is your knowledge: {self.knowledge}"
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        response_text = response.choices[0].message.content

        steps = [step.strip() for step in response_text.split("\n") if step.strip()]

        return steps
