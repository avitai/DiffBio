"""The documentation plot helpers import and label a bar chart."""

from __future__ import annotations

import matplotlib.pyplot as plt

from scripts._plot_helpers import label_bars


def test_label_bars_writes_one_label_per_bar() -> None:
    fig, ax = plt.subplots()
    bars = ax.bar([0, 1], [1.0, 2.0])

    label_bars(ax, bars, [1.0, 2.0], fmt=".1f")

    assert [text.get_text() for text in ax.texts] == ["1.0", "2.0"]
    plt.close(fig)
