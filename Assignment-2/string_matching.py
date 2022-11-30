import pandas as pd
import numpy as np
import dask.dataframe as dd
from tqdm.auto import tqdm
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct
from sklearn.feature_extraction.text import TfidfVectorizer


def main():
    amz = pd.read_csv("amz_com-ecommerce_sample.csv", encoding="latin")
    flip = pd.read_csv("flipkart_com-ecommerce_sample.csv", encoding="latin")

    amz = amz.drop_duplicates("product_name", keep="first").reset_index()
    flip = flip.drop_duplicates("product_name", keep="first").reset_index()

    flip = flip[~flip.retail_price.isna()]
    amz = amz[~amz.retail_price.isna()]

    df_amz = dd.from_pandas(amz.product_name, npartitions=6)
    df_flip = dd.from_pandas(flip.product_name, npartitions=6)

    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
    tf_idf_matrix_df_amz_clean = vectorizer.fit_transform(df_amz)
    tf_idf_matrix_df_flip_dirty = vectorizer.transform(df_flip)

    csr_matrix = awesome_cossim_top(tf_idf_matrix_df_flip_dirty, tf_idf_matrix_df_amz_clean.transpose(), 10, 0.8)

    df_matched = get_matches_df(csr_matrix, df_flip, df_amz) \
        .drop_duplicates("dirty", keep="first") \
        .reset_index().drop(0).drop("index", axis=1) \
        .drop_duplicates("clean", keep="first") \
        .reset_index().drop("index", axis=1)
    Matched_Dataset = pd.DataFrame()

    a, b, c = find_prices(df_matched=df_matched,
                          column="dirty",
                          df=flip)

    Matched_Dataset['Product name in Flipkart'] = a
    Matched_Dataset['Retail Price in Flipkart'] = b
    Matched_Dataset['Discounted Price in Flipkart'] = c

    a, b, c = find_prices(df_matched=df_matched,
                          column="clean",
                          df=amz)

    Matched_Dataset['Product name in Amazon'] = a
    Matched_Dataset['Retail Price in Amazon'] = b
    Matched_Dataset['Discounted Price in Amazon'] = c

    Matched_Dataset.to_csv("matched_dataset.csv")


def ngrams(string,
           n=3):
    """
    Returns a list of all combinations of n consecutive letters
    for a given string.
    Args:
        string (str): A given string
        n      (int): The number of letters to use
    Returns:
        list: all n letter combinations of the string
    """
    string = string.upper()
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]


def awesome_cossim_top(A, B, ntop, lower_bound=0):
    # force A and B as a CSR matrix.
    # If they have already been CSR, there is no overhead
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape

    idx_dtype = np.int32

    nnz_max = M * ntop

    indptr = np.zeros(M + 1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)

    ct.sparse_dot_topn(
        M, N, np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr, indices, data)

    return csr_matrix((data, indices, indptr), shape=(M, N))


def get_matches_df(similarity_matrix, A, B):
    '''
    Takes a matrix with similarity scores and two arrays, A and B,
    as an input and returns the matches with the score as a dataframe.
    Args:
        similarity_matrix (csr_matrix)  : The matrix (dimensions: len(A)*len(B)) with the similarity scores
        A              (pandas.Series)  : The array to be matched (dirty)
        B              (pandas.Series)  : The baseline array (clean)
    Returns:
        pandas.Dataframe : Array with matches between A and B plus scores
    '''
    non_zeros = similarity_matrix.nonzero()

    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]

    nr_matches = sparsecols.size

    dirty = np.empty([nr_matches], dtype=object)
    clean = np.empty([nr_matches], dtype=object)
    similarity = np.zeros(nr_matches)

    dirty = np.array(A)[sparserows]
    clean = np.array(B)[sparsecols]
    similarity = np.array(similarity_matrix.data)

    df_tuples = list(zip(dirty, clean, similarity))

    return pd.DataFrame(df_tuples, columns=['dirty', 'clean', 'similarity'])


def find_prices(df_matched: pd.DataFrame,
                column: str,
                df: pd.DataFrame,
                ):
    retail_price = []
    discounted_price = []
    productname = []

    for i in tqdm(df_matched[column]):
        for j, product in enumerate(df.product_name):
            if i == product:
                productname.append(product)
                retail_price.append(df.retail_price.iloc[j])
                discounted_price.append(df.discounted_price.iloc[j])
            else:
                pass
    return productname, retail_price, discounted_price


if __name__ == '__main__':
    main()
