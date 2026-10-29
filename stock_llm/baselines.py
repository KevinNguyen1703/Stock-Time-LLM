"""Floor baselines: what any model must beat on next-day return prediction."""
import numpy as np


def direction_acc(pred, y):
    """Up/down accuracy on non-flat days only (unchanged closes have no direction)."""
    m = y != 0
    return float(np.mean((pred[m] > 0) == (y[m] > 0)) * 100)


def report(name, pred, s):
    mae = float(np.mean(np.abs(pred - s.y)))
    return {"model": name, "ret_MAE": mae, "dir_acc": direction_acc(pred, s.y)}


def run_floor_baselines(train, test):
    mean_ret = train.y.mean()
    rows = [report("zero return (RW; dir = always-down)", np.zeros_like(test.y), test),
            report("train-mean return (dir = always-up)", np.full_like(test.y, mean_ret), test),
            report("momentum (sign of last return)", test.last_ret, test),
            report("mean-reversion (-last return)", -test.last_ret, test)]
    m = test.y != 0
    rows.append({"model": "(info) test: % up among non-flat / % flat days",
                 "ret_MAE": float(np.mean(~m) * 100), "dir_acc": float(np.mean(test.y[m] > 0) * 100)})
    return rows
