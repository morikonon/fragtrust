"""Expand tag macros, number citations by first appearance, append LNCS references."""

import re
import sys
from pathlib import Path

RUN = "results/runs/cnn_m_s0"
DATA = "results/data/fft75_4096_s1_v1"
MACROS = {
    "T": f"{RUN}/analysis/trust.json",
    "MET": f"{RUN}/metrics.json",
    "META": f"{RUN}/meta.json",
    "PC": f"{RUN}/per_class_test.csv",
    "EB": f"{RUN}/analysis/entropy_bins.csv",
    "CC": f"{RUN}/analysis/conformal_class_coverage.csv",
    "MAN": f"{DATA}/manifest.json",
    "LEAK": f"{DATA}/leakage.json",
    "TUNE": "results/tuning/lgbm/best.json",
}

REFS = {
    "garfinkel": "Garfinkel, S.L.: Carving contiguous and fragmented files with fast object validation. Digit. Investig. 4(Suppl.), 2–12 (2007). https://doi.org/10.1016/j.diin.2007.06.017",
    "scalpel3": "Waguespack, K., Goodwin, S., Hendrick, G., Hildebrand, S., Landaiche III, T., Li, M., McCain, J., Saizan, T., Tucker, J., Ghawaly, J.M., Richard III, G.G.: Scalpel3: A high-performance data carving architecture for recovery of fragmented files. arXiv preprint arXiv:2608.20363 (2026)",
    "fifty": "Mittal, G., Korus, P., Memon, N.: FiFTy: Large-scale file fragment type identification using convolutional neural networks. IEEE Trans. Inf. Forensics Secur. 16, 28–41 (2021). https://doi.org/10.1109/TIFS.2020.3004266",
    "fft75": "Mittal, G., Korus, P., Memon, N.: File fragment type (FFT) - 75 dataset. IEEE DataPort (2019). https://doi.org/10.21227/kfxw-8084",
    "byte2image": "Liu, W., Wang, Y., Wu, K., Yap, K.H., Chau, L.P.: A byte sequence is worth an image: CNN for file fragment classification using bit shift and n-gram embeddings. In: IEEE 5th International Conference on Artificial Intelligence Circuits and Systems (AICAS), pp. 1–5. IEEE (2023). https://doi.org/10.1109/AICAS57966.2023.10168636",
    "bytenet": "Liu, W., Wu, K., Liu, T., Wang, Y., Yap, K.H., Chau, L.P.: ByteNet: Rethinking multimedia file fragment classification through visual perspectives. IEEE Trans. Multimedia 27, 1305–1319 (2025). https://doi.org/10.1109/TMM.2024.3521830",
    "bytercnn": "Skračić, K., Petrović, J., Pale, P.: ByteRCNN: Enhancing file fragment type identification with recurrent and convolutional neural networks. IEEE Access 11, 138176–138187 (2023). https://doi.org/10.1109/ACCESS.2023.3340441",
    "ghaleb": "Ghaleb, M., Saaim, K., Felemban, M., Al-Saleh, S., Al-Mulhem, A.: File fragment classification using light-weight convolutional neural networks. IEEE Access 12, 157179–157191 (2024). https://doi.org/10.1109/ACCESS.2024.3486180",
    "horsman": "Horsman, G.: Tool testing and reliability issues in the field of digital forensics. Digit. Investig. 28, 163–175 (2019). https://doi.org/10.1016/j.diin.2019.01.009",
    "solanke": "Solanke, A.A., Biasiotti, M.A.: Digital forensics AI: Evaluating, standardizing and optimizing digital evidence mining techniques. KI – Künstliche Intelligenz 36, 143–161 (2022). https://doi.org/10.1007/s13218-022-00763-9",
    "magika": "Fratantonio, Y., Invernizzi, L., Farah, L., Thomas, K., Zhang, M., Albertini, A., Galilee, F., Metitieri, G., Cretin, J., Petit-Bianco, A., Tao, D., Bursztein, E.: Magika: AI-powered content-type detection. In: IEEE/ACM 47th International Conference on Software Engineering (ICSE), pp. 2638–2649. IEEE (2025). https://doi.org/10.1109/ICSE55347.2025.00158",
    "mimelens": "Bommarito II, M.J.: MimeLens: Position-agnostic content-type detection for binary fragments. arXiv preprint arXiv:2606.04171 (2026)",
    "guzhov": "Guzhov, A., Wirth, C.T.: Transformer-based file fragment type classification for file carving in digital forensics. In: Proceedings of the 24th European Conference on Cyber Warfare and Security (ECCWS), vol. 24(1) (2025). https://doi.org/10.34190/eccws.24.1.3552 [CHECK: pages]",
    "modico": "Anonymous: Tesserae and MoDiCo: A billion-fragment dataset and multi-branch architecture for file fragment classification. Model card, https://huggingface.co/TesseraeAnon/modico, last accessed 2026/11/27 [CITE: final venue and full text; NeurIPS 2026 per the novelty check, paper not yet available]",
    "encod": "De Gaspari, F., Hitaj, D., Pagnotta, G., De Carli, L., Mancini, L.V.: EnCoD: Distinguishing compressed and encrypted file fragments. In: Network and System Security (NSS 2020). LNCS, Springer (2020) [CHECK: volume, pages, DOI]",
    "triumvir": "De Gaspari, F., Hitaj, D., Salaris, S., Mancini, L.V.: When entropy is not enough: Multi-modal classification of encrypted and compressed data fragments. arXiv preprint arXiv:2605.31337 (2026)",
    "guo": "Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q.: On calibration of modern neural networks. In: Proceedings of the 34th International Conference on Machine Learning (ICML). PMLR, vol. 70, pp. 1321–1330 (2017)",
    "nixon": "Nixon, J., Dusenberry, M., Jerfel, G., Nguyen, T., Liu, J., Zhang, L., Tran, D.: Measuring calibration in deep learning. arXiv preprint arXiv:1904.01685 (2019)",
    "ovadia": "Ovadia, Y., Fertig, E., Ren, J., Nado, Z., Sculley, D., Nowozin, S., Dillon, J.V., Lakshminarayanan, B., Snoek, J.: Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift. In: Advances in Neural Information Processing Systems 32 (NeurIPS) (2019)",
    "ensembles": "Lakshminarayanan, B., Pritzel, A., Blundell, C.: Simple and scalable predictive uncertainty estimation using deep ensembles. In: Advances in Neural Information Processing Systems 30 (NIPS) (2017)",
    "geifman": "Geifman, Y., El-Yaniv, R.: Selective classification for deep neural networks. In: Advances in Neural Information Processing Systems 30 (NIPS) (2017)",
    "ltt": "Angelopoulos, A.N., Bates, S., Candès, E.J., Jordan, M.I., Lei, L.: Learn then test: Calibrating predictive algorithms to achieve risk control. Ann. Appl. Stat. 19(2), 1641–1662 (2025). https://doi.org/10.1214/24-AOAS1998",
    "gentle": "Angelopoulos, A.N., Bates, S.: Conformal prediction: A gentle introduction. Found. Trends Mach. Learn. 16(4), 494–591 (2023). https://doi.org/10.1561/2200000101",
    "sadinle": "Sadinle, M., Lei, J., Wasserman, L.: Least ambiguous set-valued classifiers with bounded error levels. J. Am. Stat. Assoc. (2019). https://doi.org/10.1080/01621459.2017.1395341 [CHECK: volume, pages]",
    "romano": "Romano, Y., Sesia, M., Candès, E.J.: Classification with valid and adaptive coverage. In: Advances in Neural Information Processing Systems 33 (NeurIPS) (2020)",
    "raps": "Angelopoulos, A.N., Bates, S., Malik, J., Jordan, M.I.: Uncertainty sets for image classifiers using conformal prediction. In: International Conference on Learning Representations (ICLR) (2021)",
    "ding": "Ding, T., Angelopoulos, A.N., Bates, S., Jordan, M.I., Tibshirani, R.J.: Class-conditional conformal prediction with many classes. In: Advances in Neural Information Processing Systems 36 (NeurIPS), pp. 64555–64576 (2023)",
    "tibshirani": "Tibshirani, R.J., Barber, R.F., Candès, E.J., Ramdas, A.: Conformal prediction under covariate shift. In: Advances in Neural Information Processing Systems 32 (NeurIPS) (2019)",
    "hendrycks": "Hendrycks, D., Gimpel, K.: A baseline for detecting misclassified and out-of-distribution examples in neural networks. In: International Conference on Learning Representations (ICLR) (2017)",
    "energy": "Liu, W., Wang, X., Owens, J., Li, Y.: Energy-based out-of-distribution detection. In: Advances in Neural Information Processing Systems 33 (NeurIPS) (2020)",
    "lightgbm": "Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., Liu, T.Y.: LightGBM: A highly efficient gradient boosting decision tree. In: Advances in Neural Information Processing Systems 30 (NIPS) (2017)",
}

CITE = re.compile(r"\[(@[\w]+(?:;\s*@[\w]+)*)\]")
MACRO = re.compile(r"\{\{(" + "|".join(MACROS) + r"):")


def compress(nums: list[int]) -> str:
    nums = sorted(set(nums))
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ", ".join(map(str, nums[i : j + 1])))
        i = j + 1
    return ", ".join(out)


def main(src: Path, dst: Path) -> None:
    text = MACRO.sub(lambda m: "{{" + MACROS[m.group(1)] + ":", src.read_text())
    order: dict[str, int] = {}

    def cite(match: re.Match) -> str:
        keys = [k.strip().lstrip("@") for k in match.group(1).split(";")]
        for k in keys:
            if k not in REFS:
                raise KeyError(f"unknown reference {k}")
            order.setdefault(k, len(order) + 1)
        return "[" + compress([order[k] for k in keys]) + "]"

    text = CITE.sub(cite, text)
    unused = sorted(set(REFS) - set(order))
    refs = "\n".join(f"{n}. {REFS[k]}" for k, n in sorted(order.items(), key=lambda kv: kv[1]))
    text = text.rstrip() + "\n\n## References\n\n" + refs + "\n"
    dst.write_text(text)
    print(f"{len(order)} references cited; unused: {unused}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
