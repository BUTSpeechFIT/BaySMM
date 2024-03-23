#!/usr/bin/env python3

import numpy as np
import termtables as tt

np.set_printoptions(formatter={"all": lambda x: "{:6.2f}".format(x)})


# BiLSTM-EU baseline from original MLDoc paper
# http://www.lrec-conf.org/proceedings/lrec2018/summaries/658.html
# seq: EN, DE, FR, IT, ES
BILSTM_EU = np.asarray(
    [
        [88.40, 71.83, 72.80, 60.73, 66.65],
        [71.52, 92.03, 75.45, 56.45, 75.50],
        [76.00, 78.42, 89.75, 63.70, 70.70],
        [67.15, 66.22, 65.07, 82.88, 67.07],
        [62.70, 71.05, 62.67, 57.93, 88.28],
    ]
)

# seq: EN, DE, FR, IT, ES, RU, JA, ZH
# taken from github facebookresearch/LASER/tasks/MLDoc
LASER_GH = np.asarray(
    [
        [90.73, 86.25, 78.03, 70.20, 79.30, 67.25, 60.95, 70.98],
        [80.75, 92.70, 82.83, 73.25, 79.60, 68.18, 56.80, 72.90],
        [80.08, 87.03, 90.80, 71.08, 78.40, 67.55, 53.60, 66.12],
        [74.15, 80.73, 78.35, 85.93, 82.60, 68.83, 55.15, 56.10],
        [69.58, 79.73, 75.30, 71.10, 88.75, 59.83, 59.65, 61.70],
        [72.60, 79.62, 71.28, 67.00, 68.18, 84.65, 59.23, 65.62],
        [68.45, 81.90, 67.95, 57.98, 67.95, 53.70, 85.15, 66.12],
        [77.95, 83.38, 75.83, 70.33, 78.38, 66.62, 55.25, 88.98],
    ]
)

# http://www.lrec-conf.org/proceedings/lrec2018/summaries/658.html
# seq: EN, DE, FR, IT, ES, RU, JA, ZH
MULTICCA = np.asarray(
    [
        [92.20, 81.20, 72.38, 69.38, 72.50, 60.80, 67.63, 74.73],
        [55.95, 93.70, 71.55, 63.98, 73.23, 44.83, 60.18, 55.45],
        [64.80, 53.70, 92.50, 61.15, 65.40, 40.75, 37.75, 38.35],
        [53.70, 49.20, 62.25, 85.55, 58.68, 35.38, 45.30, 32.13],
        [74.00, 55.80, 65.63, 58.35, 94.45, 45.53, 43.40, 41.63],
        [72.50, 40.30, 44.60, 42.70, 41.03, 85.65, 39.68, 42.38],
        [54.90, 52.70, 48.30, 44.33, 54.28, 40.85, 85.35, 44.78],
        [56.00, 48.70, 53.58, 47.18, 35.53, 40.45, 50.63, 87.30],
    ]
)


def convert_to_str(row, pfx="", sfx=""):

    ret_list = []
    if pfx:
        ret_list.append(pfx)

    ret_list.extend(np.array2string(row, precision=2)[1:-1].split())

    if sfx:
        ret_list.append(sfx)

    return ret_list

def get_values(scores):

    off_ixs = np.where(~np.eye(scores.shape[0], dtype=bool))
    means = np.asarray(
        [
            np.mean(scores),
            np.mean(scores[off_ixs]),
            np.mean(np.diag(scores)),
        ]
    )
    sigmas = np.asarray(
        [
            np.std(scores),
            np.std(scores[off_ixs]),
            np.std(np.diag(scores)),
        ]
    )
    return means, sigmas

def get_mean_std_row(means, sigmas):

    means = convert_to_str(means)
    sigmas = convert_to_str(sigmas)
    row = [m + f" ({s})" for m, s in zip(means, sigmas)]
    return row

def print_score_matrix_summary(scores_dict):

    table = []
    header = ["System", "Full-matrix", "Off-diag", "Diagonal"]
    for sys_name, scores in scores_dict.items():
        means, sigmas = get_values(scores)
        row = [sys_name] + get_mean_std_row(means, sigmas)
        table.append(row)
    tt.print(table, header=header, alignment='r')



def print_transfer_summary(lids, scores_dict):
    """Prints transfer summary for each language"""

    table = []
    header = ["System"] + lids
    for sys_name, scores in scores_dict.items():
        row = np.zeros(shape=(len(lids)))
        for i, lid in enumerate(lids):
            ixs = np.delete(np.arange(len(lids), dtype=int), i)
            row[i] = np.mean(scores[i, ixs])
        table.append(convert_to_str(row, pfx=sys_name))

    tt.print(table, header=header, alignment='r')


def print_full_matrix(lids, scores, stds=None):
    """Prints full matrix of results for each language"""

    table = []
    header = ["Test Lang ->"] + lids

    for i, lid in enumerate(lids):
        row = []
        means = scores[i, :]
        if stds is not None:
            sigmas = stds[i, :]
            row = [lid] + get_mean_std_row(means, sigmas)
        else:
            row = [lid] + convert_to_str(means)
        table.append(row)

    tt.print(table, header=header, alignment='r')


def print_latex_table(scores, lids, dim, impr, set_name):
    """Print results in latex table format"""

    n_rows = "r" * len(lids)

    beg = (
        r"""
\begin{table}[h!]
 \def\arraystretch{1.4}
 \centering
 \begin{tabular}{c|"""
        + n_rows
        + r"""} \toprule
 & \multicolumn{"""
        + str(len(lids))
        + r"""}{c}{\textsc{Test language}} \\ """
    )

    info_1 = r""""""
    # & & \multicolumn{5}{c}{\textsc{~146K parallel sentences}} \\ \midrule """

    print(beg)
    print(r"  & ", end=" ")
    for lid in lids:
        print(rf"\{lid}", end=" ")
        if lid != lids[-1]:
            print("& ", end="")
        else:
            print(r"\\ \midrule", end="")
    print(info_1)
    # print(r"  \multirow{" + str(len(lids)) + "}{*}{" + dim + r"}")

    end_str = (
        r"""  \bottomrule
 \end{tabular}
 \caption{Average classification accuracy on the """
        + set_name
        + r""" set. Embedding dimension is $"""
        + dim
        + r"""$.}
 \label{tab:multi}
\end{table}
    """
    )

    add_sign = lambda x: "{:.2f}".format(x) if x > 0 else "{:.2f}".format(x)
    make_bold = (
        lambda x: "\\textbf{" + "{:.2f}".format(x) + "}"
        if abs(x) > 5
        else "{:.2f}".format(x)
    )
    fmt = {"float_kind": add_sign}

    for i, lid in enumerate(lids):
        if impr != "default":
            fmt = {"float_kind": make_bold}

        print(
            r"   ",
            f"\{lid}" + r" &",
            np.array2string(scores[i, :], precision=2, formatter=fmt, separator=r" & ")[
                1:-1
            ],
            r"\\",
        )

    print(end_str)





def print_latex_table_v2(scores, stds, lids, dim, impr, set_name, std_lim):
    """Print results in latex table format"""

    n_rows = "l" * len(lids)

    beg = (
        r"""
\begin{table}[h!]
 \def\arraystretch{1}
 \centering
 \begin{tabular}{c|"""
        + n_rows
        + r"""} \toprule
 & \multicolumn{"""
        + str(len(lids))
        + r"""}{c}{\textsc{Test language}} \\ """
    )

    info_1 = r""""""
    # & & \multicolumn{5}{c}{\textsc{~146K parallel sentences}} \\ \midrule """

    print(beg)
    print(r"  & ", end=" ")
    for lid in lids:
        print(f"{lid}", end=" ")
        if lid != lids[-1]:
            print("& ", end="")
        else:
            print(r"\\ \midrule", end="")
    print(info_1)
    # print(r"  \multirow{" + str(len(lids)) + "}{*}{" + dim + r"}")

    end_str = (
        r"""  \bottomrule
 \end{tabular}
 \caption{Average classification accuracy on the """
        + set_name
        + r""" set. Embedding dimension is $"""
        + dim
        + r"""$.}
 \label{tab:multi}
\end{table}
    """
    )

    add_sign = lambda x: "{:.2f}".format(x) if x > 0 else "{:.2f}".format(x)
    make_bold = (
        lambda x: "\\textbf{" + "{:.2f}".format(x) + "}"
        if x > 0
        else "{:.2f}".format(x)
    )
    fmt = {"float_kind": add_sign}
    for i, lid in enumerate(lids):
        if impr != "default":
            fmt = {"float_kind": make_bold}

        # print(
        #     r"   ",
        #     f"\{lid}" + r" &",
        #     np.array2string(scores[i, :], precision=2, formatter=fmt, separator=r" & ")[
        #         1:-1
        #     ],
        #     r"\\",
        # )
        print(r"  ", f"{lid.upper()}" + r" &", end=" ")
        for j, _ in enumerate(scores[i, :]):
            esym = r" &"
            if j == scores.shape[1] - 1:
                esym = r" "

            if i == j:
                print(
                    r"\cellcolor{lightgray}{"
                    + "{:.1f} ({:.1f})".format(scores[i, j], stds[i, j])
                    + r"}"
                    + esym,
                    end="",
                )
            else:
                if stds[i, j] > std_lim:
                    print(
                        r"\textbf{"
                        + "{:.1f} ({:.1f})".format(scores[i, j], stds[i, j])
                        + r"}"
                        + esym,
                        end="",
                    )
                else:
                    print(
                        " {:.1f} ({:.1f})".format(scores[i, j], stds[i, j]) + esym,
                        end="",
                    )

        print(r"\\")
    print(end_str)
