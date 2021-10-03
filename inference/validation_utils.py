
def print_val_row_mean_std(res):
    row_list = []
    for k in res.keys():
        if "mean" in k:
            row_list.append(f"${round(res[k] ,4)} \pm {round(res[k.replace('mean' ,'std')] ,2)}$")
    print(" & " + " & ".join(row_list))
