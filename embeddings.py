import sentence_transformers
from sentence_transformers import SentenceTransformer , SentenceTransformerTrainer, losses, models
from tqdm import tqdm


def make_embeddings(text_data):
    """
    Taking the questions and answer part of the bioasq dataset and produce embeddings for them
    :param text_data: The dataframe that contains the questions and answers from the dataset bioasq
    :return:Two dictionaries with the format {id:(embedding, relative passages)}
    """
    # Preparing the sentence bert model
    model_name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    embeddings_questions = {}
    embeddings_answers = {}
    # Transforming each question and answer one on one into an embedding and filling the two dictionaries
    for row in tqdm(text_data.itertuples(), total=len(text_data)):
        embedding = model.encode(row.question, show_progress_bar=False, convert_to_numpy=True)
        embeddings_questions[row.id] = (embedding, row.relevant_passage_ids)
        embedding = model.encode(row.answer, show_progress_bar=False, convert_to_numpy=True)
        embeddings_answers[row.id] = (embedding, row.relevant_passage_ids)

    return embeddings_questions, embeddings_answers


def make_embeddings_corpus(text_data):
    """

    :param text_data: Τhe dataframe that contains the passages from the dataset bioasq
    :return: One dictionary with the format {id : embedding}
    """
    # Preparing the sentence bert model
    model_name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    embeddings_corpus = {}
    # Transforming each passage in an embedding and append them in a dictionary
    for row in tqdm(text_data.itertuples(), total=len(text_data)):
        embedding = model.encode(row.passage, show_progress_bar=False, convert_to_numpy=True)
        embeddings_corpus[row.id] = embedding

    return embeddings_corpus
