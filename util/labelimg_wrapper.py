"""
labelImg launcher with a runtime monkey-patch for the PyQt5 5.15+
incompatibility. The installed labelImg 1.8.6 calls
    bar.setValue(bar.value() + bar.singleStep() * units)
which produces a float on Python 3.10+ where PyQt5 5.15 strictly
requires int. This wrapper patches MainWindow.scroll_request to
coerce to int before calling setValue.

Doesn't modify the third-party labelImg install — patch is applied
at module-load time only, in this process.

Usage:
    python3 util/labelimg_wrapper.py \\
        data/labels/images \\
        data/labels/labels/classes.txt \\
        data/labels/labels
"""

import sys


def _apply_patches():
    from labelImg import labelImg as li
    orig = li.MainWindow.scroll_request

    def scroll_request_int(self, delta, orientation):
        units = -delta / (8 * 15)
        bar = self.scroll_bars[orientation]
        bar.setValue(int(bar.value() + bar.singleStep() * units))

    li.MainWindow.scroll_request = scroll_request_int


if __name__ == "__main__":
    _apply_patches()
    # labelImg.main() reads sys.argv; pass through whatever the user supplied
    from labelImg.labelImg import main
    sys.exit(main() or 0)
