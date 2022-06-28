import numpy as np
import matplotlib.pyplot as plt
head = "10x"
for str in ["amp", "prox", "miao"]:
    try:
        file = open(head+str+"step.txt", mode="r")
    except FileNotFoundError:
        continue
    min_val, mean_val, mid_val, max_val = [], [], [], []
    for line in file.readlines():
        if line[0] == '[' and line[-2] == ']':
            parsed_line = np.array(list(map(lambda x: float(x) if x != '0' and float(x) > 0.12 else np.nan, line[1:-2].split(", "))))
            print(parsed_line)
            min_val.append(np.nanmin(parsed_line))
            mean_val.append(np.nanmean(parsed_line))
            mid_val.append(np.nanmedian(parsed_line))
            max_val.append(np.nanmax(parsed_line))
    file.close()
    print("min: ",min_val)
    print("mean: ",mean_val)
    print("mid: ",mid_val)
    print("max: ",max_val)
    plt.plot(range(len(mid_val)), mean_val, label=str)
    plt.fill_between(range(len(mean_val)), min_val, max_val, alpha=0.2, label=str)
    plt.legend()
plt.savefig('exper.jpg',dpi=200)