from typing import List, Optional, Callable, Dict, Tuple

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

from scattertext.TermDocMatrix import TermDocMatrix
from scattertext.categorygrouping.characteristic_embedder_base import CategoryEmbedderABC
from scattertext.categorygrouping.rank_embedder import RankEmbedder


def cosine(a: np.array, b: np.array) -> np.array:
    return np.linalg.norm(a * b) / (np.linalg.norm(a) * np.linalg.norm(b))

def spearman_distance(a: np.array, b: np.array) -> np.array:
    return spearmanr(a, b)[0]


class CharacteristicGrouper:
    def __init__(
            self,
            corpus: TermDocMatrix,
            non_text: bool = False,
            window_size: int = 1,
            to_text: str = ' to<br/>',
            rank_embedder: Optional[CategoryEmbedderABC] = None,
            distance_measure: Callable[[np.array, np.array], float] = None
    ):
        self.corpus = corpus
        self.window_size = window_size
        self.rank_embedder = RankEmbedder() if rank_embedder is None else rank_embedder
        self.distance_measure = spearman_distance if distance_measure is None else distance_measure
        self.non_text = non_text
        self.to_text = to_text

    def group_corpus_categories(
            self,
            category_order: Optional[List[str]] = None,
            number_of_splits: int = 10,
            min_category_count: int = 1,
            max_category_count: Optional[int] = None,
    ) -> TermDocMatrix:
        new_categories, new_categories_order = self.get_new_doc_categories(
            category_order,
            number_of_splits,
            min_category_count=min_category_count,
            max_category_count=max_category_count,
        )
        return self.corpus.recategorize(new_categories)

    def get_new_doc_categories(
            self,
            category_order: Optional[List[str]] = None,
            number_of_splits: int = 10,
            min_category_count: int = 1,
            max_category_count: Optional[int] = None,
            verbose=False
    ) -> Tuple[List, List]:
        category_to_bin_dict = self.get_category_group_dict(
            category_order,
            number_of_splits,
            min_category_count=min_category_count,
            max_category_count=max_category_count,
            verbose=verbose,
        )
        new_categories = [category_to_bin_dict[cat] for cat in self.corpus.get_category_names_by_row()]

        seen = set()
        new_categories_order = [category_to_bin_dict[x] for x in category_order
                                if not (category_to_bin_dict[x] in seen or seen.add(category_to_bin_dict[x]))]
        return new_categories, new_categories_order

    def get_category_group_dict(
            self,
            category_order: List[str] = None,
            number_of_splits: int = 10,
            min_category_count: int = 1,
            max_category_count: Optional[int] = None,
            verbose=False
    ) -> Dict[str, str]:
        category_order = sorted(self._get_categories()) if category_order is None else category_order
        assert len(category_order) == self.corpus.get_num_categories()
        embedding_mat = self.rank_embedder.embed_categories(
            corpus=self.corpus,
            non_text=self.non_text
        )
        change_point_candidates = self.get_change_point_candidates(
            embeddings=embedding_mat,
            ordered_category_names=category_order,
        )
        if min_category_count == 1 and max_category_count is None:
            cat_boundaries = change_point_candidates.iloc[:number_of_splits].sort_values(by='Order')
        else:
            cat_boundaries = self.get_constrained_change_points(
                change_point_candidates=change_point_candidates,
                number_of_categories=len(category_order),
                number_of_splits=number_of_splits,
                min_category_count=min_category_count,
                max_category_count=max_category_count,
            )
        intervals = []
        interval_names = []
        last = 0
        for _, row in cat_boundaries.iterrows():
            intervals.append((last, int(row.Order)))
            interval_names.append(
                str(category_order[last + 1 if last > 0 else 0])
                + self.to_text + str(category_order[int(row.Order)]))
            last = int(row.Order)
        intervals.append((last, len(category_order)))
        interval_names.append(str(category_order[last + 1])
                              + self.to_text + str(category_order[-1]))
        interval_index = pd.IntervalIndex.from_tuples(intervals)
        category_to_bin_dict = {category: interval_names[interval_index.get_loc(i) if i > 0 else 0]
                                for i, category in enumerate(category_order)}
        return category_to_bin_dict

    def get_constrained_change_points(
            self,
            change_point_candidates: pd.DataFrame,
            number_of_categories: int,
            number_of_splits: int,
            min_category_count: int = 1,
            max_category_count: Optional[int] = None,
    ) -> pd.DataFrame:
        if min_category_count < 1:
            raise ValueError('min_category_count must be at least 1')
        if max_category_count is not None and max_category_count < min_category_count:
            raise ValueError('max_category_count must be greater than or equal to min_category_count')
        if number_of_categories < min_category_count:
            raise ValueError(
                'number_of_categories must be greater than or equal to min_category_count'
            )

        boundaries = []
        if max_category_count is not None:
            while max(self.get_group_sizes(boundaries, number_of_categories)) > max_category_count:
                if len(boundaries) == number_of_splits:
                    raise ValueError(
                        'Could not satisfy max_category_count with the requested number_of_splits'
                    )
                accepted_boundary = None
                for order in change_point_candidates['Order'].astype(int):
                    if order <= 0 or order >= number_of_categories - 1 or order in boundaries:
                        continue
                    proposed_boundaries = sorted(boundaries + [order])
                    group_sizes = self.get_group_sizes(proposed_boundaries, number_of_categories)
                    if min(group_sizes) < min_category_count:
                        continue
                    if max(group_sizes) >= max(self.get_group_sizes(boundaries, number_of_categories)):
                        continue
                    accepted_boundary = order
                    break
                if accepted_boundary is None:
                    raise ValueError(
                        'Could not find change points satisfying min_category_count and max_category_count'
                    )
                boundaries = sorted(boundaries + [accepted_boundary])

        for order in change_point_candidates['Order'].astype(int):
            if len(boundaries) == number_of_splits:
                break
            if order <= 0 or order >= number_of_categories - 1 or order in boundaries:
                continue
            proposed_boundaries = sorted(boundaries + [order])
            group_sizes = self.get_group_sizes(proposed_boundaries, number_of_categories)
            if min(group_sizes) < min_category_count:
                continue
            boundaries = proposed_boundaries

        return change_point_candidates[
            change_point_candidates['Order'].astype(int).isin(boundaries)
        ].sort_values(by='Order')

    @staticmethod
    def get_group_sizes(
            boundaries: List[int],
            number_of_categories: int,
    ) -> List[int]:
        last = -1
        group_sizes = []
        for boundary in boundaries + [number_of_categories - 1]:
            group_sizes.append(boundary - last)
            last = boundary
        return group_sizes

    def _get_string_categories(self) -> List[str]:
        return [str(c) for c in self.corpus.get_categories()]
    def _get_categories(self) -> List[str]:
        return self.corpus.get_categories()

    def get_change_point_candidates(
            self,
            embeddings: np.array,
            ordered_category_names: List[str],
    ) -> pd.DataFrame:

        category_index_df = pd.DataFrame({
            'Category': self._get_string_categories(),
        }).reset_index().set_index('Category').loc[
            [str(x) for x in ordered_category_names]
        ].assign(
            Order=lambda df: np.arange(len(df))
        )

        mat = embeddings[:, category_index_df['index'].values]

        dists = []
        for i in range(mat.shape[1] - self.window_size):
            a = mat[:, list(range(max(0, i - self.window_size + 1), i + 1))].mean(axis=1)
            b = mat[:, list(range(i + 1, max(i + 1 + self.window_size, mat.shape[1])))].mean(axis=1)
            distance = self.distance_measure(a, b)
            dists.append(distance)

        category_index_df = category_index_df.assign(
            Dist=dists + [0] * self.window_size,
        ).sort_values(by='Dist', ascending=True)
        return category_index_df
