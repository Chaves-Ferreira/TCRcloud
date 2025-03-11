import sys
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from natsort import natsort_keygen

import tcrcloud.format
import tcrcloud.colours
import tcrcloud.vlength

# Import default dicts that contain V gene information
TRAV = tcrcloud.colours.TRAV
TRBV = tcrcloud.colours.TRBV
TRGV = tcrcloud.colours.TRGV
TRDV = tcrcloud.colours.TRDV
IGHV = tcrcloud.colours.IGHV
IGKV = tcrcloud.colours.IGKV
IGLV = tcrcloud.colours.IGLV

def get_table(keys, samples, args):
    if args.compare.lower() != "true":
        if args.compare.lower() != "false":
            sys.stderr.write("TCRcloud error: please indicate True or False\n")
            exit()

    datasets = []
    for_comparison = {"A": [], "B": [], "G": [], "D": [], "H": [], "K": [], "L": []}
    for j in keys:
        if j[0] == "A":
            x_axis = TRAV
            plot_aspect = (3.5, 1, 1)
            x_size = 6
            ymax = args.yhighalpha
            ymin = args.ylowalpha
            zmax = args.zhighalpha
            zmin = args.zlowalpha
        if j[0] == "B":
            x_axis = TRBV
            plot_aspect = (3, 1, 1)
            x_size = 4
            ymax = args.yhighbeta
            ymin = args.ylowbeta
            zmax = args.zhighbeta
            zmin = args.zlowbeta
        if j[0] == "G":
            x_axis = TRGV
            plot_aspect = (2, 1, 1)
            x_size = 10
            ymax = args.yhighgamma
            ymin = args.ylowgamma
            zmax = args.zhighgamma
            zmin = args.zlowgamma
        if j[0] == "D":
            x_axis = TRDV
            plot_aspect = (1.5, 1, 1)
            x_size = 10
            ymax = args.yhighdelta
            ymin = args.ylowdelta
            zmax = args.zhighdelta
            zmin = args.zlowdelta
        if j[0] == "H":
            x_axis = IGHV
            plot_aspect = (5, 1, 1)
            x_size = 2
            ymax = args.yhighheavy
            ymin = args.ylowheavy
            zmax = args.zhighheavy
            zmin = args.zlowheavy
        if j[0] == "K":
            x_axis = IGKV
            plot_aspect = (3.5, 1, 1)
            x_size = 2
            ymax = args.yhighkappa
            ymin = args.ylowkappa
            zmax = args.zhighkappa
            zmin = args.zlowkappa
        if j[0] == "L":
            x_axis = IGLV
            plot_aspect = (3.5, 1, 1)
            x_size = 4
            ymax = args.yhighlambda
            ymin = args.ylowlambda
            zmax = args.zhighlambda
            zmin = args.zlowlambda

        x_axis_ticks = list(range(len(x_axis)))
        df = samples.get_group(j)

        new_df = df.pivot_table(index=["v_call", "CDR3_length"], aggfunc="size").reset_index()
        new_df.rename(columns={0: "counts"}, inplace=True)

        # Create an empty df to serve as base
        empty_df = pd.DataFrame(columns=["v_call", "CDR3_length", "counts"])
        x_axis_names = []
        for v_genes, colour in x_axis.items():
            x_axis_names.append(v_genes)
            if ymin is None:
                limitymin = new_df.loc[new_df["CDR3_length"].idxmin()].iloc[1] - 1
            else:
                limitymin = ymin + 1
            if ymax is None:
                limitymax = new_df.loc[new_df["CDR3_length"].idxmax()].iloc[1] + 1
            else:
                limitymax = ymax - 1
            for c in range(limitymin, limitymax):
                df_new_row = pd.DataFrame({"v_call": [v_genes], "CDR3_length": [c], "counts": [0]})
                empty_df = pd.concat([empty_df, df_new_row])

        df_merged = pd.concat([new_df, empty_df], ignore_index=True, sort=True)
        final_df = df_merged.groupby(["v_call", "CDR3_length"]).sum().reset_index()
        final_df["frequency"] = 100 * (final_df["counts"] / final_df["counts"].sum())
        final_df = final_df.sort_values(by=["CDR3_length", "v_call"], key=natsort_keygen())
        df_grouped = final_df.groupby(["v_call", "CDR3_length"]).sum()
        df_reset = df_grouped.reset_index()
        df_reformat = df_reset.pivot(index="v_call", columns="CDR3_length", values="frequency").reset_index().rename_axis(index=None, columns=None)
        df_sorted = df_reformat.sort_values(by=["v_call"], key=natsort_keygen())

        if args.export.lower() in ["true"]:
            df_filename = args.rearrangements[:-4] + "_vgenes_table" + j[1] + "_" + j[0] + ".csv"
            df_sorted.to_csv(df_filename, index=False)

        x = df_sorted["v_call"].factorize()[0]
        y = np.array(df_sorted.columns.values.tolist()[1:])
        df_transpose = df_sorted.transpose()
        z = np.array(df_transpose.values.tolist()[1])
        for i in range(2, len(df_sorted.columns)):
            z = np.append(z, np.array(df_transpose.values.tolist()[i]))
        z = np.array_split(z, len(df_sorted.columns) - 1)

        datasets.append([
            x, y, z, plot_aspect, x_size, ymin, ymax, zmin, zmax,
            x_axis_ticks, x_axis_names, j[0], j[1], False,
        ])
        for_comparison[j[0]].append([
            df_sorted, plot_aspect, x_size, ymin, ymax, zmin, zmax,
            x_axis_ticks, x_axis_names, j[0] + "_comparison", j[1],
        ])

    if args.compare.lower() == "false":
        return datasets
    elif args.compare.lower() == "true":
        comparisons = []
        for m in for_comparison:
            if len(for_comparison[m]) < 2:
                sys.stderr.write("Less than 2 repertoires from the " + m + " chain were detected in the rearragements file\n")
            if len(for_comparison[m]) > 2:
                sys.stderr.write("More than 2 repertoires from the " + m + " chain were detected in the rearragements file\n")
            if len(for_comparison[m]) == 2:
                comb = for_comparison[m]
                num1 = comb[0][0].copy().drop("v_call", axis=1)
                num2 = comb[1][0].copy().drop("v_call", axis=1)
                # Append infer_objects() after fillna to avoid FutureWarning
                comparison1 = num1 - num2
                comparison1 = comparison1.fillna(0).infer_objects()
                comparison2 = num2 - num1
                comparison2 = comparison2.fillna(0).infer_objects()
                comparison1.insert(0, "v_call", comb[0][0]["v_call"])
                comparison2.insert(0, "v_call", comb[0][0]["v_call"])

                if comb[0][3] is not None:
                    ymin = min(comb[0][3], comb[1][3])
                else:
                    ymin = None
                if comb[0][4] is not None:
                    ymax = max(comb[0][4], comb[1][4])
                else:
                    ymax = None
                if comb[0][5] is not None:
                    zmin = max(comb[0][5], comb[1][5])
                else:
                    zmin = None
                if comb[0][6] is not None:
                    zmax = max(comb[0][6], comb[1][6])
                else:
                    zmax = None

                x = comparison1["v_call"].factorize()[0]
                y = np.array(comparison1.columns.values.tolist()[1:])
                df_transpose = comparison1.transpose()
                z = np.array(df_transpose.values.tolist()[1])
                for i in range(2, len(comparison1.columns)):
                    z = np.append(z, np.array(df_transpose.values.tolist()[i]))
                z = np.array_split(z, len(comparison1.columns) - 1)

                comparisons.append([
                    x, y, z, comb[0][1], comb[0][2], ymin, ymax, zmin, zmax,
                    comb[0][7], comb[0][8], comb[0][9], comb[0][10], True,
                ])

                x = comparison2["v_call"].factorize()[0]
                y = np.array(comparison2.columns.values.tolist()[1:])
                df_transpose = comparison2.transpose()
                z = np.array(df_transpose.values.tolist()[1])
                for i in range(2, len(comparison2.columns)):
                    z = np.append(z, np.array(df_transpose.values.tolist()[i]))
                z = np.array_split(z, len(comparison2.columns) - 1)

                comparisons.append([
                    x, y, z, comb[1][1], comb[1][2], ymin, ymax, zmin, zmax,
                    comb[1][7], comb[1][8], comb[1][9], comb[1][10], True,
                ])
        return comparisons

def generate_spectratyping_plots(args, formatted_samples):
    # Process each chain (A, B, G, D, H, K, L)
    chains = formatted_samples['chain'].unique()
    colour_dicts = {
        "A": TRAV, "B": TRBV, "G": TRGV, "D": TRDV,
        "H": IGHV, "K": IGKV, "L": IGLV
    }

    for chain in chains:
        df_chain = formatted_samples[formatted_samples["chain"] == chain]
        if df_chain.empty:
            continue

        # Build a pivot table: rows = v_call, columns = CDR3_length, fill missing with 0
        pivot_df = df_chain.pivot_table(
            index="v_call",
            columns="CDR3_length",
            aggfunc="size",
            fill_value=0
        )

        # Reindex rows to include all calls from the chain dictionary (even absent ones)
        chain_colours = colour_dicts.get(chain, {})
        all_calls_for_chain = list(chain_colours.keys())  # preserve dictionary order
        pivot_df = pivot_df.reindex(all_calls_for_chain, fill_value=0)

        # Convert counts to frequencies
        total = pivot_df.values.sum()
        freq_df = pivot_df * 100 / total if total > 0 else pivot_df

        # --------------------------
        # 1) Comparison Mode
        # --------------------------
        if args.compare.lower() == "true":
            # Must have exactly 2 repertoires to compare
            rep_groups = df_chain.groupby("repertoire_id")
            if len(rep_groups) != 2:
                sys.stderr.write(
                    f"Chain {chain} in compare mode requires exactly 2 repertoires. Skipping chain.\n"
                )
                continue

            # Build freq tables per repertoire
            freq_dfs = {}
            for rep, df_rep in rep_groups:
                pivot_rep = df_rep.pivot_table(
                    index="v_call", columns="CDR3_length", aggfunc="size", fill_value=0
                )
                # Reindex so every call from the chain dictionary appears
                pivot_rep = pivot_rep.reindex(all_calls_for_chain, fill_value=0)
                total_rep = pivot_rep.values.sum()
                freq_dfs[rep] = pivot_rep * 100 / total_rep if total_rep > 0 else pivot_rep

            # Union of all V calls
            union_v_calls = all_calls_for_chain  # same order as dictionary
                
            n_plots = len(union_v_calls)
            n_cols = int(np.ceil(np.sqrt(n_plots)))
            n_rows = int(np.ceil(n_plots / n_cols))
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
            axs = axs.flatten()

            # Determine a global y‑axis max across all V calls & both reps
            all_values = []
            for rep_df in freq_dfs.values():
                all_values.extend(rep_df.values.flatten())
            y_max = max(all_values) if len(all_values) > 0 else 0

            # We’ll define two contrasting colors
            compare_colors = ["#1f77b4", "#d62728"]
            reps = list(freq_dfs.keys())
            width = 0.4  # bar width for side‑by‑side

            # For each V call, create one subplot
            for i, v_call in enumerate(union_v_calls):
                ax = axs[i]

                # For each repertoire, draw bars side by side
                for j, rep in enumerate(reps):
                    # Get the frequency row for this v_call (or zero if missing)
                    df_rep = freq_dfs[rep]
                    row = df_rep.loc[v_call]
                    # Ensure the x_values are the union of all lengths in df_rep.columns
                    x_values = np.array(sorted(df_rep.columns))
                    y_values = row.reindex(x_values, fill_value=0).values

                    # Offset each bar to the left or right
                    offset = (-width / 2) if j == 0 else (width / 2)
                    ax.bar(
                        x_values + offset,
                        y_values,
                        width=width,
                        color=compare_colors[j],
                        label=f"Rep {rep}" if i == 0 else ""  # legend label only for top row
                    )

                ax.set_title(v_call)
                ax.set_xlabel("CDR3 Length")
                ax.set_ylabel("% of reads")
                # Set the same y‑range across all subplots
                ax.set_ylim([0, y_max * 1.1])

                # Integer ticks for each CDR3 length
                ax.set_xticks(x_values)
                ax.set_xticklabels([str(int(x)) for x in x_values], rotation=45)

            # Hide any unused subplots
            for j in range(n_plots, len(axs)):
                axs[j].axis("off")

            # Create a single legend for the entire figure (top-right)
            handles, labels = axs[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper right")

            plt.suptitle(f"Spectratyping Compare Mode - Chain {chain}")
            plt.tight_layout()
            plt.subplots_adjust(top=0.96)
            outputname = args.rearrangements[:-4] + f"_spectratyping_compare_chain_{chain}.png"
            plt.savefig(outputname)
            print(f"Spectratyping compare image saved as {outputname}")

        # --------------------------
        # 2) Non‑Comparison Mode
        # --------------------------
        else:
            v_calls = all_calls_for_chain  # same order as dictionary
            n_plots = len(v_calls)
            n_cols = int(np.ceil(np.sqrt(n_plots)))
            n_rows = int(np.ceil(n_plots / n_cols))
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
            axs = axs.flatten()

            # Determine a global y‑axis max
            y_max = freq_df.values.max() if freq_df.size > 0 else 0

            for i, v_call in enumerate(v_calls):
                ax = axs[i]
                # Reindex the row so it matches the full set of CDR3 lengths
                row = freq_df.loc[v_call]
                x_values = np.array(sorted(freq_df.columns), dtype=float)
                y_values = row.reindex(x_values, fill_value=0).values

                # Use the colour from the dictionary; default to grey if not found
                colour = chain_colours.get(v_call, "#808080")

                # Make a bar plot
                ax.bar(x_values, y_values, color=colour)

                ax.set_title(v_call)
                ax.set_xlabel("CDR3 Length")
                ax.set_ylabel("% of reads")

                # Same Y range for all subplots
                ax.set_ylim([0, y_max * 1.1])

                # Integer ticks for all CDR3 lengths
                ax.set_xticks(x_values)
                ax.set_xticklabels([str(int(x)) for x in x_values], rotation=45)

            # Hide any unused subplots
            for j in range(n_plots, len(axs)):
                axs[j].axis("off")
            plt.suptitle(f"Spectratyping - Chain {chain}")
            plt.tight_layout()
            plt.subplots_adjust(top=0.96)
            outputname = args.rearrangements[:-4] + f"_spectratyping_chain_{chain}.png"
            plt.savefig(outputname)
            print(f"Spectratyping image saved as {outputname}")

def generate_vj_heatmap(args, formatted_samples):
    # Get J-gene dictionary from tcrcloud.colours
    j_gene_dicts = {
        "A": tcrcloud.colours.TRAJ,
        "B": tcrcloud.colours.TRBJ,
        "G": tcrcloud.colours.TRGJ,
        "D": tcrcloud.colours.TRDJ,
        "H": tcrcloud.colours.IGHJ,
        "K": tcrcloud.colours.IGKJ,
        "L": tcrcloud.colours.IGLJ
    }

    v_gene_dicts = {
        "A": tcrcloud.colours.TRAV,
        "B": tcrcloud.colours.TRBV,
        "G": tcrcloud.colours.TRGV,
        "D": tcrcloud.colours.TRDV,
        "H": tcrcloud.colours.IGHV,
        "K": tcrcloud.colours.IGKV,
        "L": tcrcloud.colours.IGLV
    }

    # Process each chain separately
    chains = formatted_samples["chain"].unique()

    for chain in chains:
        df_chain = formatted_samples[formatted_samples["chain"] == chain]
        if df_chain.empty:
            continue

        # Group by repertoire_id (in case comparison is flagged)
        rep_groups = df_chain.groupby("repertoire_id")

        # Get corresponding V and J genes dictionary and sort it
        v_gene_set = v_gene_dicts.get(chain, set())
        j_gene_set = j_gene_dicts.get(chain, set())
        # Ensure V and J genes columns are correctly ordered
        all_v_genes = sorted(v_gene_set)
        all_j_genes = sorted(j_gene_set)

        # --------------------------------------------------
        # 1) Comparison Mode
        # --------------------------------------------------
        if args.compare.lower() == "true":
            if len(rep_groups) != 2:
                sys.stderr.write(
                    f"Chain {chain} in compare mode requires exactly 2 repertoires. Skipping chain.\n"
                )
                continue

            # Extract each repertoire’s data
            rep_keys = list(rep_groups.groups.keys())  # e.g. ["rep1", "rep2"]
            rep1, rep2 = rep_keys[0], rep_keys[1]

            df_rep1 = rep_groups.get_group(rep1)
            df_rep2 = rep_groups.get_group(rep2)

            # Build pivot tables: rows=V, cols=J
            pivot1 = df_rep1.pivot_table(
                index="v_call", columns="j_call", aggfunc="size", fill_value=0
            )
            pivot2 = df_rep2.pivot_table(
                index="v_call", columns="j_call", aggfunc="size", fill_value=0
            )

            # Reindex so every V and J gene is present
            pivot1 = pivot1.reindex(index=all_v_genes, columns=all_j_genes, fill_value=0)
            pivot2 = pivot2.reindex(index=all_v_genes, columns=all_j_genes, fill_value=0)

            # Convert to relative frequencies
            total1 = pivot1.values.sum()
            total2 = pivot2.values.sum()
            freq1 = (pivot1 * 100 / total1) if total1 > 0 else pivot1
            freq2 = (pivot2 * 100 / total2) if total2 > 0 else pivot2

            # Create two difference DataFrames: (rep1 - rep2) and (rep2 - rep1)
            diff1 = freq1 - freq2  # If positive => rep1 higher
            diff2 = freq2 - freq1  # If positive => rep2 higher

            # Define a diverging colormap with center at 0
            cmap = plt.cm.RdBu_r  # red for +, blue for -, white near 0

            # Set figure size and resolution dynamically
            if chain == "A":  # TRAJ has 61 entries, needs higher resolution
                fig_width, fig_height, dpi = 14, 10, 400
                x_fontsize = 7  # Reduce font size for x-axis labels
            elif chain in ["B", "K", "L", "H"]:  # Chains with crowded Y-axis
                fig_width, fig_height, dpi = 10, 16, 400  # Increase height
                x_fontsize = 10
            else:
                fig_width, fig_height, dpi = 10, 10, 300
                x_fontsize = 10  # Normal font size for smaller chains

            # ------------- Plot diff1 (rep1 - rep2) -------------
            plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
            data_min = diff1.values.min()
            data_max = diff1.values.max()
            # Center the color scale at 0
            norm = mcolors.TwoSlopeNorm(vmin=data_min, vcenter=0, vmax=data_max)

            plt.imshow(diff1, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
            plt.colorbar(label="(Rep1 - Rep2) % Difference")
            plt.xlabel(f"{chain}-J Calls", fontsize=x_fontsize)
            plt.ylabel(f"{chain}-V Calls", fontsize=10)
            plt.title(f"VJ Pairing Heatmap Difference ({chain} Chain)\n{rep1} - {rep2}")

            # Set tick labels
            plt.xticks(ticks=range(len(all_j_genes)), labels=all_j_genes, rotation=45, fontsize=x_fontsize)
            plt.yticks(ticks=range(len(all_v_genes)), labels=all_v_genes, fontsize=10)

            plt.tight_layout()
            outname1 = args.rearrangements[:-4] + f"_VJ_heatmap_{chain}_{rep1}_minus_{rep2}.png"
            plt.savefig(outname1, dpi=dpi)
            print(f"VJ difference heatmap saved as {outname1}")

            # ------------- Plot diff2 (rep2 - rep1) -------------
            plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
            data_min = diff2.values.min()
            data_max = diff2.values.max()
            # Center the color scale at 0
            norm = mcolors.TwoSlopeNorm(vmin=data_min, vcenter=0, vmax=data_max)

            plt.imshow(diff2, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
            plt.colorbar(label="(Rep2 - Rep1) % Difference")
            plt.xlabel(f"{chain}-J Calls")
            plt.ylabel(f"{chain}-V Calls")
            plt.title(f"VJ Pairing Heatmap Difference ({chain} Chain)\n{rep2} - {rep1}")

            # Set tick labels
            plt.xticks(ticks=range(len(all_j_genes)), labels=all_j_genes, rotation=45, fontsize=x_fontsize)
            plt.yticks(ticks=range(len(all_v_genes)), labels=all_v_genes, fontsize=10)

            plt.tight_layout()
            outname2 = args.rearrangements[:-4] + f"_VJ_heatmap_{chain}_{rep2}_minus_{rep1}.png"
            plt.savefig(outname2)
            print(f"VJ difference heatmap saved as {outname2}")

        # --------------------------------------------------
        # 2) Non-Comparison Mode
        # --------------------------------------------------
        else:
            # Normal single heatmap: rows = V genes, columns = J genes
            pivot_df = df_chain.pivot_table(
                index="v_call",
                columns="j_call",
                aggfunc="size",
                fill_value=0
            )

            # Ensure all expected V and J genes are included, even if absent in the data
            pivot_df = pivot_df.reindex(index=all_v_genes, columns=all_j_genes, fill_value=0)

            # Normalize counts to percentages
            total = pivot_df.values.sum()
            heatmap_data = (pivot_df * 100) / total if total > 0 else pivot_df

            # Set figure size and resolution dynamically
            if chain == "A":  # TRAJ has 61 entries, needs higher resolution
                fig_width, fig_height, dpi = 14, 10, 400
                x_fontsize = 7  # Reduce font size for x-axis labels
            elif chain in ["B", "K", "L", "H"]:  # Chains with crowded Y-axis
                fig_width, fig_height, dpi = 10, 16, 400  # Increase height
                x_fontsize = 10
            else:
                fig_width, fig_height, dpi = 10, 10, 300
                x_fontsize = 10  # Normal font size for smaller chains

            # Generate heatmap
            plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
            # Use `"RdYlBu"` colormap
            cmap = plt.cm.RdYlBu_r  # Reverse so red = high values
            norm = mcolors.Normalize(vmin=heatmap_data.min().min(), vmax=heatmap_data.max().max())

            # # to Restore the original colormap (Yellow = High, Blue = Low) change above 2 rows for:
            # cmap = plt.cm.viridis  # Original colormap
            # norm = mcolors.Normalize(vmin=heatmap_data.min().min(), vmax=heatmap_data.max().max())

            plt.imshow(heatmap_data, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
            plt.colorbar(label="Relative Percentage")
            plt.xlabel(f"{chain}-J Calls")
            plt.ylabel(f"{chain}-V Calls")
            plt.title(f"VJ Pairing Heatmap ({chain} Chain)")

            # Set tick labels
            plt.xticks(ticks=range(len(pivot_df.columns)), labels=pivot_df.columns, rotation=45, fontsize=x_fontsize)
            plt.yticks(ticks=range(len(pivot_df.index)), labels=pivot_df.index, fontsize=10)

            plt.tight_layout()
            outputname = args.rearrangements[:-4] + f"_VJ_heatmap_{chain}.png"
            plt.savefig(outputname, dpi=dpi)
            print(f"VJ heatmap saved as {outputname}")

def barplot(args):
    samples_df = tcrcloud.format.format_data(args)
    formatted_samples = tcrcloud.format.format_vgene(samples_df)

    # Generate spectratyping plots if the flag is set
    if getattr(args, "spectratyping", False):
        generate_spectratyping_plots(args, formatted_samples)

    # Generate VJ pairing heatmap if the flag is set
    if getattr(args, "vj", False):
        generate_vj_heatmap(args, formatted_samples)

    # Continue with existing 3D surface plots
    samples = formatted_samples.groupby(["chain", "repertoire_id"])
    keys = [key for key, _ in samples]
    datasets = get_table(keys, samples, args)

    for i in datasets:
        if i[13] is False:
            fig = go.Figure(go.Surface(
                x=i[0], y=i[1], z=i[2], colorscale="Turbo", cmin=i[7], cmax=i[8]
            ))
            camera = dict(eye=dict(x=2.5, y=-3.5, z=2.5))
            sc = dict(
                aspectratio=dict(x=i[3][0], y=i[3][1], z=i[3][2]),
                xaxis_title=i[10][0][:4],
                yaxis_title="CDR3 Length",
                zaxis_title="Percentage of reads",
                xaxis=dict(
                    tickmode="array",
                    ticktext=i[10],
                    tickvals=i[9],
                    tickfont=dict(size=i[4]),
                    titlefont=dict(size=10),
                ),
                yaxis=dict(tickfont=dict(size=8), titlefont=dict(size=10), range=[i[5], i[6]]),
                zaxis=dict(tickfont=dict(size=8), titlefont=dict(size=10), range=[i[7], i[8]]),
            )
            fig.update_layout(
                width=700,
                margin=dict(r=10, l=10, b=10, t=10),
                scene_camera=camera,
                scene=sc,
                template="plotly_white",
            )
            outputname = args.rearrangements[:-4] + "_vgenes_" + i[12] + "_" + i[11] + ".png"
            fig.write_image(outputname, scale=6)
            print("V genes plot saved as " + outputname)

        else:
            i = datasets[0]
            fig = go.Figure(go.Surface(
                x=i[0], y=i[1], z=i[2], colorscale="Portland", cmin=i[7], cmax=i[8]
            ))
            camera = dict(eye=dict(x=2.5, y=-5, z=0.5))
            sc = dict(
                aspectratio=dict(x=i[3][0], y=i[3][1], z=i[3][2]),
                xaxis_title=i[10][0][:4],
                yaxis_title="CDR3 Length",
                zaxis_title="Percentage of reads",
                xaxis=dict(
                    tickmode="array",
                    ticktext=i[10],
                    tickvals=i[9],
                    tickfont=dict(size=i[4]),
                    tickangle=-45,
                    titlefont=dict(size=10),
                ),
                yaxis=dict(tickfont=dict(size=6), titlefont=dict(size=10), range=[i[5], i[6]]),
                zaxis=dict(tickfont=dict(size=8), titlefont=dict(size=10), range=[i[7], i[8]]),
            )
            fig.update_layout(
                width=700,
                margin=dict(r=10, l=10, b=10, t=10),
                scene_camera=camera,
                scene=sc,
                template="plotly_white",
            )
            outputname = args.rearrangements[:-4] + "_vgenes_" + i[12] + "_" + i[11] + ".png"
            fig.write_image(outputname, scale=6)
            print("V genes plot saved as " + outputname)

            i = datasets[1]
            fig = go.Figure(go.Surface(
                x=i[0], y=i[1], z=i[2], colorscale="Portland_r", cmin=i[7], cmax=i[8]
            ))
            camera = dict(eye=dict(x=2.5, y=-5, z=0.5))
            sc = dict(
                aspectratio=dict(x=i[3][0], y=i[3][1], z=i[3][2]),
                xaxis_title=i[10][0][:4],
                yaxis_title="CDR3 Length",
                zaxis_title="Percentage of reads",
                xaxis=dict(
                    tickmode="array",
                    ticktext=i[10],
                    tickvals=i[9],
                    tickfont=dict(size=i[4]),
                    tickangle=-45,
                    titlefont=dict(size=10),
                ),
                yaxis=dict(tickfont=dict(size=6), titlefont=dict(size=10), range=[i[5], i[6]]),
                zaxis=dict(tickfont=dict(size=8), titlefont=dict(size=10), range=[i[7], i[8]]),
            )
            fig.update_layout(
                width=700,
                margin=dict(r=10, l=10, b=10, t=10),
                scene_camera=camera,
                scene=sc,
                template="plotly_white",
            )
            outputname = args.rearrangements[:-4] + "_vgenes_" + i[12] + "_" + i[11] + ".png"
            fig.write_image(outputname, scale=6)
            print("V genes plot saved as " + outputname)
            break

    if args.export.lower() in ["true"]:
        output_folder = os.getcwd()
        try:
            tcrcloud.vlength.process_csv_files(output_folder)
            print("vlength analysis completed and saved in the 'analysis' subfolder.")
        except Exception as e:
            print(f"An error occurred in vlength processing: {e}")
