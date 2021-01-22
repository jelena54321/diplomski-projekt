import argparse
from collections import namedtuple
import pysam
import itertools
from abc import ABC
from abc import abstractmethod
from coder import Coder
import gen

class Region:
    """
    A class that represents a single sequence region.

    Attributes
    ----------
    name : region name, i.e. ID of the corresponding reference
    start : region start
    end : region end
    """

    def __init__(self, name, start, end):
        self.name = name
        self.start = start
        self.end = end

class TargetAlign:
    """
    A class that represents a single align.

    Attributes
    ----------
    align : aligned segment
    start : align start
    end : align end
    """

    def __init__(self, align, start, end):
        self.align = align
        self.start = start
        self.end = end

class AlignedPosition:
    """
    A class that represents a single aligned position.

    Attributes
    ----------
    query_position : position on an aligned query
    query_base : query base on the aligned position
    ref_position : position on the reference genome
    ref_base : reference base on the aligned position
    """

    def __init__(self, query_position, query_base, ref_position, ref_base):
        self.query_position = query_position
        self.query_base = query_base
        self.ref_position = ref_position
        self.ref_base = ref_base

WINDOW = 100_000
OVERLAP = 300

def generate_regions(ref, ref_name, window=WINDOW, overlap=OVERLAP):
    """
    Generates regions for the provided sequence.

    Parameters
    ----------
    ref : sequence that need to be devided into regions
    ref_name : corresponding sequence name
    window : size of a single region
    overlap : size of a window overlap

    Returns
    -------
    regions : generated regions
    """

    length = len(ref)
    i = 0
    while i < length:
        end = i + window
        yield Region(name=ref_name, start=i, end=min(end, length))

        if end >= length: break
        i = end - overlap

def generate_inference_data(args):
    """
    Generates inference data for the region provided through arguments.

    Parameters
    ----------
    reads_path : path to the aligned reads file
    ref : reference sequence
    region : region for which data is required

    Returns
    -------
    region_name : region name
    positions : positions corresponding provided region
    examples : examples corresponding provided region
    """

    reads_path, ref, region = args

    region_string = f'{region.name}:{region.start + 1}-{region.end}'
    result = gen.generate_features(reads_path, ref, region_string)

    positions = []
    examples = []
    for P, X in zip(*result):
        positions.append(P)
        examples.append(X)

    print(f'>> finished generating examples for {region.name}:{region.start}-{region.end}')
    return region.name, positions, examples

REF_START_GETTER = lambda r: r.align.reference_start
REF_LEN_GETTER = lambda r: r.align.reference_length
ALIGN_START_GETTER = lambda a: a.start

def generate_train_data(args):
    """
    Generates train data for the region provided through arguments.

    Parameters
    ----------
    reads_path : path to the aligned reads file
    truth_genome_path : path to the truth genome
    ref : reference sequence
    region : region for which data is required

    Returns
    -------
    region_name : region name
    positions : positions corresponding provided region
    examples : examples corresponding provided region
    labels : labels corresponding provided region
    """

    reads_path, truth_genome_path, ref, region = args

    aligns = get_aligns(truth_genome_path, region)
    filtered_aligns = filter_aligns(aligns)

    print(f'>> finished generating labels for {region.name}:{region.start}-{region.end}')

    if not filtered_aligns: 
        print(f'>> no alignments')
        return None

    positions = []
    examples = []
    labels = []

    for align in filtered_aligns:
        position_label_dict = dict()
        positions_with_unknown_base = set()

        pos, lbls = get_postions_and_labels(align, ref, region)
        for position, label in zip(pos, lbls):
            if label == Coder.encode(Coder.UNKNOWN):
                positions_with_unknown_base.add(position)
            else:
                position_label_dict[position] = label

        sorted_positions = sorted(list(position_label_dict.keys()))
        region_string = f'{region.name}:{sorted_positions[0][0] + 1}-{sorted_positions[-1][0]}'
        result = gen.generate_features(reads_path, str(ref), region_string)

        for P, X in zip(*result):
            Y = []
            to_yield = True

            for p in P:
                assert is_in_region(p[0], filtered_aligns)

                if p in positions_with_unknown_base:
                    to_yield = False
                    break

                try:
                    y_label = position_label_dict[p]
                except KeyError:
                    if p[1] != 0:
                        y_label = Coder.encode(Coder.GAP)
                    else:
                        raise KeyError(f'error: No label mapping for position {p}!')

                Y.append(y_label)

            if to_yield:
                positions.append(P)
                examples.append(X)
                labels.append(Y)

    print(f'>> finished generating examples for {region.name}:{region.start}-{region.end}')
    return region.name, positions, examples, labels

def get_aligns(truth_genome_path, region):
    """
    Returns truth genome aligns corresponding the provided region.

    Parameters
    ----------
    truth_genome_path : path to the truth genome file
    region : region for which the aligns are required
    """

    aligns = []
    with pysam.AlignmentFile(truth_genome_path, 'rb', index_filename=truth_genome_path + '.bai') as f:
        for r in f.fetch(region.name, region.start, region.end):
            if r.reference_name != region.name: raise ValueError()
            if r.reference_end <= region.start or r.reference_start >= region.end: continue

            if not r.is_unmapped and not r.is_secondary:
                aligns.append(TargetAlign(align=r, start=r.reference_start, end=r.reference_end))

    aligns.sort(key=REF_START_GETTER)
    return aligns

def filter_aligns(aligns, len_threshold=2.0, overlap_threshold=0.5, min_len=1000):
    """
    Filters aligns that satisfy certain prerequisites.

    Firstly, aligns are ordered by reference start and afterwards by their length.
    Secondly, length ratio (ratio between longer and shorter align) and overlap
    ratio (ration between overlap and short align length) is calculated.

    In case:
    - where length ratio is smaller than the length thereshold:
        - if overlap ratio is smaller than overlap threshold:
            end of the first align is now overlap start 
            and
            start of second align is now overlap end
        - else
            both aligns are discarded
    - where length ratio is bigger or equal to length ratio
        - if overlap ratio is equal or bigger than overlap ratio
            shorter align is discarded
        - else
            start of second align is now overlap end

    Finally, only those aligns that are now longer than the minimal required
    length are returned.

    Returns
    -------
    filtered_aligns : aligns that satisfy certain conditions.
    """

    to_be_removed = set()
    for i, j in itertools.combinations(aligns, 2):
        first, second = order_by_ref_start(i, j)

        overlap = get_overlap(first, second)
        if overlap is None: continue
        overlap_start, overlap_end = overlap

        shorter, longer = order_by_ref_len(i, j)

        len_ratio = longer.align.reference_length / shorter.align.reference_length
        overlap_ratio = (overlap_end - overlap_start) / shorter.align.reference_length

        if len_ratio < len_threshold:
            if overlap_ratio < overlap_threshold:
                first.end = overlap_start
                second.start = overlap_end
            else:
                to_be_removed.add(shorter)
                to_be_removed.add(longer)

        else:
            if overlap_ratio >= overlap_threshold:
                to_be_removed.add(shorter)
            else:
                second.start = overlap_end

    filtered_aligns = list(filter(lambda a: (a.end - a.start >= min_len) and a not in to_be_removed, aligns))
    filtered_aligns.sort(key=ALIGN_START_GETTER)
    return filtered_aligns

def get_overlap(first, second):
    """
    Gets overlap for provided aligs.

    Parameters
    ----------
    first : first align
    second : second align
    """

    if second.start < first.end:
        return second.start, first.end
    else:
        return None

def order_by_ref_start(first, second):
    """
    Returns aligns ordered by reference start.

    Parameters
    ----------
    first : first align
    second : second align
    """

    return sorted((first, second), key=REF_START_GETTER)

def order_by_ref_len(first, second):
    """
    Returns aligns ordered by length.

    Parameters
    ----------
    first : first align
    second : second align
    """

    return sorted((first, second), key=REF_LEN_GETTER)

def get_pairs(align, ref):
    """
    Gets aligned positions for provided align and reference sequence.

    Parameters
    ----------
    align : read aligned to the reference sequence
    ref : reference sequence
    """

    query = align.query_sequence
    if query is None: raise StopIteration()

    for query_position, ref_position in align.get_aligned_pairs():
        ref_base = ref[ref_position] if ref_position is not None else None
        query_base = query[query_position] if query_position is not None else None
        yield AlignedPosition(query_position, query_base, ref_position, ref_base)

def get_postions_and_labels(align, ref, region):
    """
    Returns list of corresponding positions and labels.

    Parameters
    ----------
    align : align for which positions and labels are required
    ref : corresponding reference sequence
    region : corresponding region
    """

    start, end = region.start, region.end
    if start is None: start = 0
    if end is None: end = float('inf')
    start, end = max(start, align.start), min(end, align.end)

    positions = []
    labels = []

    pairs = get_pairs(align.align, ref)
    current_position = None
    insert_count = 0

    for pair in itertools.dropwhile(lambda p: (p.ref_position is None) or (p.ref_position < start), pairs):
        if pair.ref_position == align.align.reference_end or (pair.ref_position is not None and pair.ref_position >= end):
            break

        if pair.ref_position is None:
            insert_count += 1
        else:
            insert_count = 0
            current_position = pair.ref_position

        position = (current_position, insert_count)
        positions.append(position)

        label = pair.query_base.upper() if pair.query_base else Coder.GAP

        try:
            encoded_label = Coder.encode(label)
        except KeyError:
            encoded_label = Coder.encode(Coder.UNKNOWN)

        labels.append(encoded_label)

    return positions, labels

def is_in_region(position, aligns):
    """
    Returns true if at least one of the provided aligns contains
    provided postions.

    Parameters
    ----------
    position : sequence position
    aligns : list of aligns
    """

    for align in aligns:
        if align.start <= position < align.end: return True
    return False
