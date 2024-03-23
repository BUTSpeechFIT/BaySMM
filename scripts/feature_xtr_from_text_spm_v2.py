#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract bag-of-words counts from given text
"""

import re
import os
import sys
import csv
import json
import codecs
import pickle
import string
import argparse
from typing import Tuple
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import sentencepiece as sp
from pylibs.nlp.textvector import BoW
from pylibs.misc.io import read_simple_flist


#
# https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
#
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def save_as_chunks(dbyw, out_base_name: str, chunk_size: int):
    """Save BoW stats in chunks"""

    chunk_fnames = []

    one_chunk_only = False
    six = 0
    eix = six + chunk_size
    if eix > dbyw.shape[0]:
        eix = dbyw.shape[0]
        one_chunk_only = True

    i = 1
    while six < dbyw.shape[0]:
        if one_chunk_only:
            out_f = f"{out_base_name}.npz"
        else:
            out_f = f"{out_base_name}_chunk_{i}.npz"

        sparse.save_npz(out_f, dbyw[six:eix, :])
        chunk_fnames.append(out_f)

        # print("{:7d} {:7d}".format(six, eix), out_f, "saved.")

        six = eix
        eix += chunk_size
        if eix > dbyw.shape[0]:
            eix = dbyw.shape[0]
        i += 1

    return chunk_fnames


def remove_punc(data: str):
    """Remove punctuation. Input data is string"""
    translator = str.maketrans({key: None for key in string.punctuation})
    clean_data = data.translate(translator)
    clean_data = re.sub("\s\s+", " ", clean_data)  # remove multiple spaces
    return clean_data


def jap_tokenizer(sent):
    """Japanese tokenizer"""

    try:
        tokens = JAP_TOK.tokenize(sent)
    except UnicodeEncodeError as err:
        print("- Set MECAB_CHARSET=utf8", file=sys.stderr)
        sys.exit()

    return [str(t) for t in tokens]


def zh_tokenizer(sent):
    """Chinese tokenizer"""

    words = ZH_TOK.cut(sent)
    return [w.strip() for w in words if w.strip()]


def load_sentences(
    input_text,
    input_type,
    topk=-1,
    topr=1.0,
    silent: bool = False,
) -> Tuple[list, dict]:
    """Load sentences given the input text or flist"""

    name2pos = (
        {}
    )  # file basename to start-end position in concatenated list of sentences

    if input_type == "flist":
        files = read_simple_flist(input_text)
    else:
        files = [input_text]

    sentences = []

    offset = 0
    for i, fname in enumerate(files):
        basename, ext = os.path.basename(fname).rsplit(".", 1)
        if not silent:
            print(" {:3d}/{:3d} | {:s} ".format(i + 1, len(files), fname))
        lno = 0
        cur_sents = []
        with open(fname, "r", encoding="utf-8") as fpr:
            for line in fpr:
                cur_sents.append(line.strip())
                lno += 1
                if topk != -1 and lno == topk:
                    break

        toprk = int(topr * lno)
        sentences.extend(cur_sents[:toprk])

        name2pos[basename] = [offset, offset + toprk]
        offset += toprk

        print(" -", basename, name2pos[basename])

    print(" - # Sentences  :", len(sentences))

    return sentences, name2pos


def extract_bow_features(sentences, name2pos, args):
    """Extract counts (bag-of-words) for the give data"""

    print("- Building vocabulary and extracting BoW stats..")

    if args.mv:
        max_vocab = int(args.mv)
    else:
        max_vocab = None

    strip = "unicode"
    lower = True
    if args.nostrip:
        strip = None
    if args.nolower:
        lower = False

    vocab = None
    if args.vocab_file:
        vocab = []
        ext = os.path.basename(args.vocab_file).rsplit(".", 1)[-1]
        if ext == "json":
            vocab_d = {}
            with open(args.vocab_file, "r", encoding="utf-8") as fpr:
                vocab_d = json.load(fpr)
            int2word = {}
            for word, idx in vocab_d.items():
                int2word[idx] = word
            vocab = []
            for i in range(len(int2word)):
                vocab.append(int2word[i])
        else:
            vocab = read_simple_flist(args.vocab_file)
        print(" - In vocabulary:", len(vocab))
        args.mdf = 1
        max_vocab = None

    print(" - Analyzer     :", args.ana)
    print(" - N-gram range :", args.ng)
    print(" - Strip accents:", strip)
    print(" - Lowecase     :", lower)
    print(" - Min. doc.freq:", args.mdf)
    print(" - Max. vocab.  :", max_vocab)

    if args.lang in ("zh", "zh_cn"):
        c_vec = CountVectorizer(
            ngram_range=tuple(args.ng),
            strip_accents=strip,
            tokenizer=zh_tokenizer,
            min_df=args.mdf,
            analyzer=args.ana,
            max_features=max_vocab,
            vocabulary=vocab,
        )

    elif args.lang == "ja":
        c_vec = CountVectorizer(
            ngram_range=tuple(args.ng),
            strip_accents=strip,
            decode_error="ignore",
            tokenizer=jap_tokenizer,
            min_df=args.mdf,
            analyzer=args.ana,
            max_features=max_vocab,
            vocabulary=vocab,
        )
    else:
        c_vec = CountVectorizer(
            ngram_range=tuple(args.ng),
            strip_accents=strip,
            min_df=args.mdf,
            analyzer=args.ana,
            max_features=max_vocab,
            lowercase=lower,
            vocabulary=vocab,
        )

    if args.ng[0] == args.ng[1]:
        sfx = f"{args.ana}_n{args.ng[0]}_mdf_{args.mdf}_mv_{args.mv}"
    else:
        sfx = f"{args.ana}_n{args.ng[0]}-{args.ng[1]}_mdf_{args.mdf}_mv_{args.mv}"
    if args.topk > 0:
        sfx += f"_topk_{args.topk}"

    f_count_vect = os.path.join(args.out_dir, f"{args.lang}_cvect_{sfx}.pkl")
    if os.path.exists(f_count_vect):
        print("- CountVectorizer:", f_count_vect, "already exists.")
        sys.exit()

    dbyw = c_vec.fit_transform(sentences)
    dbyw_f = os.path.join(args.out_dir, f"{args.lang}_counts_{sfx}.npz")
    sparse.save_npz(dbyw_f, dbyw)
    print("- BoW stats size:", dbyw.shape)

    for basename, pos in name2pos.items():
        out_par_dir = os.path.join(args.out_dir, f"{args.lang}/parallel")
        os.makedirs(out_par_dir, exist_ok=True)
        out_file = os.path.join(out_par_dir, f"{basename}_counts_{sfx}.npz")
        if os.path.exists(out_file):
            print("- File already exists.", out_file)
            continue
        else:
            sparse.save_npz(out_file, dbyw[pos[0] : pos[1], :])
            print("- Saved", out_file, dbyw[pos[0] : pos[1], :].shape)

    with open(f_count_vect, "wb") as fpr:
        pickle.dump(c_vec, fpr)

    vocab_f = os.path.join(args.out_dir, f"{args.lang}_vocab_{sfx}.json")
    with codecs.open(vocab_f, "w", "utf-8") as fpw:
        json.dump(
            c_vec.vocabulary_,
            fpw,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            cls=NumpyEncoder,
        )

    if args.extract_list:
        for i, elist in enumerate(args.extract_list):
            extract_files = read_simple_flist(elist)
            print(
                "- BoW stats will be extracted for",
                len(extract_files),
                "files using the obtained vocabulary of size:",
                len(c_vec.vocabulary_),
            )
            for afile in extract_files:
                base = ""
                if args.base_dir:
                    base += os.path.basename(os.path.dirname(afile)) + "_"
                base += os.path.basename(afile).rsplit(".", 1)[0]
                print(" -", base, end="")
                data, _ = load_sentences(
                    afile,
                    "text",
                    silent=True,
                )
                bow_e = c_vec.transform(data)
                # print(bow_e.shape)
                out_f = os.path.join(args.xtr_out_dir[i], f"{base}_counts_{sfx}.npz")

                # if bow_e.shape[0] > args.chunk_size and args.chunk_size > 0:
                # print("  - Will also be saved in chunks of size:", args.chunk_size)
                save_as_chunks(
                    bow_e,
                    os.path.join(args.xtr_out_dir[i], f"{base}_counts_{sfx}"),
                    args.chunk_size,
                )
                # else:
                #    sparse.save_npz(out_f, bow_e)

    print(
        "- Sparse counts (npz) matrices, vocabulary and pkl files saved in",
        args.out_dir,
    )


def main():
    """main"""

    args = parse_arguments()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.xtr_out_dir:
        for xtr_dir in args.xtr_out_dir:
            os.makedirs(xtr_dir, exist_ok=True)

    print("- Loading text ..")
    if args.topk > 0:
        print(" - Will pick only top", args.topk, "sentences from each file.")
    if args.topr > 0:
        print(" - Will pick only top", args.topr, "ratio of sentences from each file.")

    sentences, name2pos = load_sentences(
        args.text_f,
        args.input_type,
        topk=args.topk,
        topr=args.topr,
        silent=(not args.verbose),
    )

    if args.tokenizer == "spm":
        model_f = os.path.join(args.out_dir, f"{args.lang}_spm_{args.vocab_size}.model")

        if not os.path.exists(model_f):
            sp.SentencePieceTrainer.train(
                input=args.text_f,
                model_prefix=args.out_dir + f"{args.lang}_spm_{args.vocab_size}",
                vocab_size=args.vocab_size,
            )

        print("- Sentencepiece model trained:", model_f)

        sp_encoder = sp.SentencePieceProcessor(model_file=model_f)
        # enc_sentences = [sp_encoder.encode_as_pieces(line) for line in sentences]
        # sentences = enc_sentences
        enc_sentences = sp_encoder.encode(sentences)
        i_sentences = [" ".join([str(i) for i in l]) for l in enc_sentences]

        sfx = f"{args.vocab_size}"

        vocab = []
        for id in range(sp_encoder.get_piece_size()):
            vocab.append(sp_encoder.id_to_piece(id))

        bow = BoW(lower=False, remove_punc=None, analyzer="word", ngram=1)
        dbyw = bow.fit_transform(i_sentences)

        bow_vocab_f = os.path.join(args.out_dir, f"{args.lang}_bow_vocab.json")
        with open(bow_vocab_f, "w", encoding="utf-8") as fpw:
            json.dump(bow.vocab_d, fpw, indent=2, ensure_ascii=False, sort_keys=True)
        print(bow_vocab_f, "saved.")

        out_f = os.path.join(args.out_dir, f"{args.lang}_{args.vocab_size}.npz")
        sparse.save_npz(out_f, dbyw)
        print("- Saved", out_f)

        if args.extract_list:
            extract_files = read_simple_flist(args.extract_list)
            for afile in extract_files:
                base = os.path.basename(afile).rsplit(".", 1)[0]
                data = load_sentences(
                    afile, "text", args.out_dir, silent=(not args.verbose)
                )
                bow_e = bow.transform(data)
                print(afile, bow_e.shape)

                out_f = os.path.join(args.xtr_out_dir, f"{base}_counts_{sfx}.npz")

                sparse.save_npz(out_f, bow_e)
                print(out_f, "saved.")

        f_count_vect = os.path.join(args.out_dir, f"{args.lang}_bow_{sfx}.pkl")
        # if args.lang != "zho":
        pickle.dump(bow, open(f_count_vect, "wb"))

        vocab_f = os.path.join(args.out_dir, f"{args.lang}_vocab_{sfx}.txt")
        with codecs.open(vocab_f, "w", "utf-8") as fpw:
            fpw.write("\n".join(vocab) + "\n")

        print(
            "- Sparse counts (npz) matrices, vocabulary and pkl files saved in",
            args.out_dir,
        )

    else:
        extract_bow_features(sentences, name2pos, args)


def parse_arguments():
    """parse arguments"""

    global JAP_TOK, ZH_TOK

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("text_f", help="path to a text file or flist")
    parser.add_argument("input_type", choices=["text", "flist"], help="input type")
    parser.add_argument(
        "lang",
        choices=[
            "en",
            "de",
            "fr",
            "it",
            "es",
            "ru",
            "zh",
            "zh_cn",
            "pl",
            "ja",
            "ar",
            "ka",
            "rw",
            "te",
            "kn",
            "ur",
            "ta",
            "tr",
            "or",
            "ml",
            "mr",
            "bn",
            "gu",
            "hi",
            "pa",
        ],
        help="ISO 639-1 language code\
 (https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)",
    )
    parser.add_argument("out_dir", help="output dir")
    parser.add_argument(
        "-tokenizer",
        default="normal",
        type=str,
        choices=["normal", "spm"],
        help="choice of tokenizer.\
     normal: white space tokenizer or jieba for zh or konoha(mecab) for ja \
     spm: defaults to unigram in sentencepiece",
    )
    parser.add_argument(
        "-vocab_size",
        type=int,
        default=16000,
        help="vocab size in case of sentencepiece (unigram)",
    )

    parser.add_argument(
        "-extract_list",
        type=str,
        nargs="+",
        help="extract features for the list of files",
    )
    parser.add_argument(
        "-xtr_out_dir",
        type=str,
        nargs="+",
        help="a different out dir for the extract list?",
    )

    parser.add_argument(
        "-ana",
        default="word",
        choices=["word", "char_wb"],
        help="analyzer for count vectorizer or BoW",
    )
    parser.add_argument(
        "-ng",
        type=int,
        nargs=2,
        default=(1, 1),
        choices=[1, 2, 3],
        help="ngram range of tokens",
    )
    parser.add_argument(
        "-mdf", type=int, default=1, help="min. doc freq. constraint on vocabulary"
    )
    parser.add_argument(
        "-vocab_file",
        default="",
        help="input vocab file to use. Has precedence over -mv and -mdf",
    )
    parser.add_argument(
        "-mv",
        type=int,
        default=100000,
        help="max vocab size in case of bag-of-words extracted using sklearn CountVectorizer",
    )

    subset_group = parser.add_mutually_exclusive_group()

    subset_group.add_argument(
        "-topr",
        type=float,
        default=1.0,
        choices=np.arange(1, 11, 1) / 10.0,
        help="ratio of number of sentences per flist to be considered. Useful in preparing training sets of different sizes.",
    )

    subset_group.add_argument(
        "-topk",
        type=int,
        default=-1,
        help="pick only top K sentences from unlabelled parallel text. \
If input_type is flist then top K will be picked for each file in the flist. In total it will be K * len(flist) sentences.",
    )
    # subset_group.add_argument(
    #    "-subset_ixs", type=str, default="", help="file with line indices"
    # )

    parser.add_argument("-header", default=None, help="header for input csv file")
    parser.add_argument(
        "--nolower",
        default=False,
        action="store_true",
        help="Do not lowercase the text",
    )
    parser.add_argument(
        "--nostrip", default=False, action="store_true", help="Do not strip accents"
    )
    parser.add_argument(
        "--base_dir",
        action="store_true",
        help="append base_dir and prefix to out file names (useful when extract_list contains same base file names in different sub dirs)",
    )
    parser.add_argument(
        "-chunk_size",
        type=int,
        default=200000,
        help="save large files in xtr_flist in chunks for easy loading while training the multilingual model.",
    )

    parser.add_argument("--verbose", action="store_true", help="increased verbosity")

    args = parser.parse_args()

    if args.lang == "ja":
        from konoha import WordTokenizer

        JAP_TOK = WordTokenizer("MeCab")

    elif args.lang in ("zh", "zh_cn"):
        import jieba

        ZH_TOK = jieba.Tokenizer(dictionary=jieba.DEFAULT_DICT)

    if len(args.extract_list) != len(args.xtr_out_dir):
        print(
            "- Input args error: Number of args for -extract_list ({:d}) should match the number \
of args for -xtr_out_dir ({:d})".format(
                len(args.extract_list), len(args.xtr_out_dir)
            ),
            file=sys.stderr,
        )
        sys.exit()

    return args


if __name__ == "__main__":
    main()
