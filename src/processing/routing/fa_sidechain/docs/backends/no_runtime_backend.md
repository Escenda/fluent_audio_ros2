# no_runtime_backend

`fa_sidechain` は外部 runtime backend を持たない。

RMS 計算、threshold 判定、dB から linear gain への変換は node 内の決定的な処理で完結する。model load、device access、network call、worker process は使わない。

この package が publish するのは control stream であり、program audio の加工は下流 node の責務である。
