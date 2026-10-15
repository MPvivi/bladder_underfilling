# -*- coding: utf-8 -*-
"""
bladder_underfilling.py

Code for "Simulation-based assessment of bladder wall DVH robustness to simulated
bladder underfilling in prostate VMAT using a single planning CT".
Repository: https://github.com/MPvivi/bladder_underfilling

For ONE patient, this script
  1. reads the planning RTSTRUCT and RTDOSE (DICOM),
  2. rasterises the bladder and prostate contours on the RTDOSE grid,
  3. simulates bladder underfilling by displacing the dome caudally
     (primary model, Section 2.B of the paper),
  4. derives the 4-mm bladder wall at each level and computes its DVH on the
     unchanged planned dose,
  5. reports V70Gy, V60Gy and V40Gy (% and cc) for each level, and the
     last level within limits and the first level exceeding a limit.

Google Colab
  Upload the RTDOSE and RTSTRUCT files of one patient to /content (or a sub-folder),
  then run
      !pip install -q pydicom scikit-image
      !python bladder_underfilling.py --dicom_dir /content --bladder Bladder --prostate Prostate
  or open bladder_underfilling_colab.ipynb.

Command line
  python bladder_underfilling.py --dicom_dir path/to/patient --out results
  python bladder_underfilling.py --rtdose RD.dcm --rtstruct RS.dcm --out results

Outputs in --out
  levels.csv      one row per simulated level (and the planning contour)
  dvh.csv         cumulative wall DVH for every level (long format)
  summary.txt     last level within limits / first level exceeding a limit
  dvh.png         wall DVHs with the analytical reference limits
  deformation.png sagittal view of the planning and simulated bladder on the planned dose
                  (CT is used as background if CT files are in the same folder)

The planned dose is not recalculated; the dose distribution is assumed to stay
fixed relative to the prostate. No patient data are included with this code.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import pydicom
from scipy import ndimage
from skimage.draw import polygon2mask

# ------------------------------------------------------------------ settings
D_MAX_MM = 30.0                                   # maximum nominal dome displacement
STEP_MM = 3.0                                     # step of the nominal level
WALL_MM = 4.0                                     # wall thickness
FIXED_MM = dict(anterior=4.0, posterior=2.0, lateral=3.0)   # primary model, applied at every level
SMOOTH_MM = 2.0                                   # final dilation (smoothing of step artefacts)
LIMITS = {70: 0.25, 60: 0.35, 40: 0.50}           # analytical reference limits (fraction of wall)
REPORT_DOSES = (70, 60, 40)


# ------------------------------------------------------------------ DICOM input
class Case:
    """Dose grid and structure masks of one patient (all on the RTDOSE grid)."""

    def __init__(self, rd_path, rs_path, bladder_name, prostate_name):
        ds = pydicom.dcmread(rd_path, force=True)
        self.dose = ds.pixel_array.astype(np.float32) * float(ds.DoseGridScaling)
        self.dx, self.dy = float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])
        gfov = np.array(ds.GridFrameOffsetVector, dtype=float)
        if len(gfov) > 1 and gfov[1] < gfov[0]:          # make the slice index increase cranially
            self.dose, gfov = self.dose[::-1].copy(), gfov[::-1].copy()
        self.dz = float(gfov[1] - gfov[0])
        self.voxel_size_mm = (self.dz, self.dy, self.dx)
        self.ipp = np.array(ds.ImagePositionPatient, dtype=float)
        iop = np.array(ds.ImageOrientationPatient, dtype=float)
        self.row_dir, self.col_dir = iop[0:3], iop[3:6]
        self.z = self.ipp[2] + gfov
        self.Nz, self.Ny, self.Nx = self.dose.shape
        self.vox_cc = abs(self.dx * self.dy * self.dz) / 1000.0
        self.rs = pydicom.dcmread(rs_path, force=True)
        self.bladder = self.mask(bladder_name)
        self.prostate = self.mask(prostate_name)
        kk = np.where(self.bladder)[0]
        # the bladder base cannot move caudally: keep only slices from the most
        # inferior slice of the planning bladder upwards
        self.clip = np.zeros_like(self.bladder)
        self.clip[kk.min():] = True
        self.sag_i = int(round(np.where(self.bladder)[2].mean()))   # sagittal plane through the bladder centre

    def _w2v(self, p):
        rel = np.asarray(p, float) - self.ipp
        i = np.dot(rel, self.row_dir) / self.dx
        j = np.dot(rel, self.col_dir) / self.dy
        k = int(np.argmin(np.abs(self.z - p[2])))
        return k, j, i

    def mask(self, roi_name):
        rois = {r.ROIName: r.ROINumber for r in self.rs.StructureSetROISequence}
        match = [n for n in rois if n.strip().lower() == roi_name.strip().lower()]
        if not match:
            raise ValueError(f"Structure '{roi_name}' not found. Available: {sorted(rois)}")
        rc = [r for r in self.rs.ROIContourSequence if r.ReferencedROINumber == rois[match[0]]][0]
        m = np.zeros(self.dose.shape, dtype=bool)
        for c in rc.ContourSequence:
            pts = np.array(c.ContourData, float).reshape(-1, 3)
            kji = np.array([self._w2v(p) for p in pts])
            k = int(round(kji[:, 0].mean()))
            if 0 <= k < self.Nz:
                m[k] |= polygon2mask((self.Ny, self.Nx), np.stack([kji[:, 1], kji[:, 2]], axis=1))
        return m


def find_dicom(dicom_dir):
    """Return the RTDOSE (plan dose preferred) and RTSTRUCT files found in dicom_dir."""
    rd, rs = [], []
    for f in sorted(glob.glob(os.path.join(dicom_dir, "**", "*"), recursive=True)):
        if not os.path.isfile(f):
            continue
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
        except Exception:
            continue
        mod = getattr(ds, "Modality", "")
        if mod == "RTDOSE":
            rd.append((getattr(ds, "DoseSummationType", "") != "PLAN", f))
        elif mod == "RTSTRUCT":
            rs.append(f)
    if not rd or not rs:
        raise FileNotFoundError(f"RTDOSE and RTSTRUCT are needed in {dicom_dir} "
                                f"(found {len(rd)} RTDOSE, {len(rs)} RTSTRUCT).")
    if len(rs) > 1:
        print(f"Several RTSTRUCT files found; using {rs[0]}")
    return sorted(rd)[0][1], rs[0]


# ------------------------------------------------------------------ morphology
def shift(mask, sk, sj, si):
    """Integer voxel shift without wrap-around."""
    Nz, Ny, Nx = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    k0, k1 = max(0, sk), Nz + min(0, sk); j0, j1 = max(0, sj), Ny + min(0, sj); i0, i1 = max(0, si), Nx + min(0, si)
    a0, a1 = max(0, -sk), Nz - max(0, sk); b0, b1 = max(0, -sj), Ny - max(0, sj); c0, c1 = max(0, -si), Nx - max(0, si)
    out[k0:k1, j0:j1, i0:i1] = mask[a0:a1, b0:b1, c0:c1]
    return out


UNIT = {"cranial": [0, 0, 1.], "anterior": [0, -1., 0], "posterior": [0, 1., 0],
        "left": [1., 0, 0], "right": [-1., 0, 0]}


def vec_to_shift(C, vec):
    dz, dy, dx = C.voxel_size_mm
    k = int(round(-vec[2] / dz))
    ip = np.array(vec, float) - np.array([0, 0, vec[2]])
    return k, int(round(np.dot(ip, C.col_dir) / dy)), int(round(np.dot(ip, C.row_dir) / dx))


def cut(C, m, direction, d_mm):
    """Intersect the mask with a copy shifted by d_mm (moves one surface inward)."""
    if d_mm <= 0:
        return m
    return m & shift(m, *vec_to_shift(C, np.array(UNIT[direction]) * d_mm))


def morph(C, mask, dist_mm, mode):
    """6-connected dilation/erosion by round(dist / largest voxel size) voxels."""
    n = int(round(abs(dist_mm) / max(C.voxel_size_mm)))
    if n <= 0:
        return mask.copy()
    st = ndimage.iterate_structure(ndimage.generate_binary_structure(3, 1), n)
    return ndimage.binary_dilation(mask, st) if mode == "expand" else ndimage.binary_erosion(mask, st)


def simulate_bladder(C, d_mm):
    """Simulated bladder (primary model) for nominal dome displacement d_mm.

    The dome is displaced caudally by d_mm; small fixed in-plane components
    (FIXED_MM) are applied at every level, followed by 2-mm smoothing. The
    bladder base is fixed (no voxel below the most inferior planning slice).
    """
    m = cut(C, C.bladder, "cranial", d_mm)
    m = cut(C, m, "anterior", FIXED_MM["anterior"])
    m = cut(C, m, "posterior", FIXED_MM["posterior"])
    m = cut(C, m, "left", FIXED_MM["lateral"])
    m = cut(C, m, "right", FIXED_MM["lateral"])
    m = morph(C, m, SMOOTH_MM, "expand")
    return m & C.clip


def wall(C, m, t=WALL_MM):
    """4-mm wall: mask minus its 6-connected erosion, excluding the prostate."""
    return m & ~morph(C, m, t, "shrink") & ~C.prostate


# ------------------------------------------------------------------ DVH
def dvh(C, w, bw=0.1):
    vals = C.dose[w]
    bins = np.arange(0.0, float(vals.max()) + bw, bw)
    h, e = np.histogram(vals, bins=bins)
    vcc = h[::-1].cumsum()[::-1] * C.vox_cc
    return pd.DataFrame({"dose_Gy": e[:-1], "volume_frac": vcc / vcc[0], "volume_cc": vcc})


def vx(d, x):
    return float(d.volume_frac.iloc[(d.dose_Gy - x).abs().idxmin()])


def metrics(C, m, label, d_mm):
    w = wall(C, m)
    d = dvh(C, w)
    wcc = w.sum() * C.vox_cc
    top0 = C.z[np.where(C.bladder)[0]].max()
    rec = {"level": label, "d_nominal_mm": d_mm,
           "dome_descent_eff_mm": top0 - C.z[np.where(m)[0]].max(),
           "bladder_cc": m.sum() * C.vox_cc, "wall_cc": wcc}
    for x in REPORT_DOSES:
        rec[f"V{x}Gy_pct"] = 100 * vx(d, x)
        rec[f"V{x}Gy_cc"] = vx(d, x) * wcc
    rec["limits_exceeded"] = "+".join(f"V{x}Gy" for x, lim in LIMITS.items() if vx(d, x) > lim)
    return rec, d.assign(level=label)


# ------------------------------------------------------------------ main
def run(dicom_dir=None, rtdose=None, rtstruct=None, bladder="Bladder", prostate="Prostate",
        step=STEP_MM, out="results", show=False):
    """Analyse one patient. Give dicom_dir, or rtdose and rtstruct. Returns the levels table."""
    if rtdose is None or rtstruct is None:
        rtdose, rtstruct = find_dicom(dicom_dir or ".")
    print(f"RTDOSE  : {rtdose}\nRTSTRUCT: {rtstruct}")
    os.makedirs(out, exist_ok=True)
    a = argparse.Namespace(step=step, out=out)

    C = Case(rtdose, rtstruct, bladder, prostate)
    print(f"Dose grid {C.dx:g} x {C.dy:g} x {abs(C.dz):g} mm; planning bladder {C.bladder.sum() * C.vox_cc:.1f} cc")

    rows, dvhs, masks = [], [], {"planning contour": C.bladder}
    r, d = metrics(C, C.bladder, "planning contour", np.nan)
    rows.append(r); dvhs.append(d)
    for dmm in np.round(np.arange(0.0, D_MAX_MM + 1e-6, a.step), 3):
        m = simulate_bladder(C, dmm)
        if not m.any():
            break
        r, d = metrics(C, m, f"{dmm:g} mm", dmm)
        rows.append(r); dvhs.append(d); masks[f"{dmm:g} mm"] = m
    L = pd.DataFrame(rows)
    L.to_csv(os.path.join(a.out, "levels.csv"), index=False, float_format="%.6g")
    pd.concat(dvhs).to_csv(os.path.join(a.out, "dvh.csv"), index=False, float_format="%.5g")

    # endpoint: last level within limits and first level exceeding a limit (simulated levels only)
    S = L[L.level != "planning contour"].reset_index(drop=True)
    hit = S.index[S.limits_exceeded != ""]
    lines = [f"Primary model, {a.step:g}-mm steps, limits "
             + ", ".join(f"V{x}Gy <= {100 * v:g}%" for x, v in LIMITS.items())]
    c0 = L.iloc[0]
    lines.append(f"Planning contour: {c0.bladder_cc:.1f} cc; limits exceeded: {c0.limits_exceeded or 'none'}")
    if len(hit) == 0:
        lines.append(f"No limit exceeded within {S.d_nominal_mm.max():g} mm "
                     f"(bladder {S.bladder_cc.iloc[-1]:.1f} cc at the last level).")
        first = last = None
    else:
        first = S.iloc[hit[0]]
        last = S.iloc[hit[0] - 1] if hit[0] > 0 else None
        if last is not None:
            lines.append(f"Last level within limits: d = {last.d_nominal_mm:g} mm "
                         f"(effective {last.dome_descent_eff_mm:g} mm), bladder {last.bladder_cc:.1f} cc")
        else:
            lines.append("A limit is already exceeded at the 0-mm level.")
        lines.append(f"First level exceeding a limit: d = {first.d_nominal_mm:g} mm "
                     f"(effective {first.dome_descent_eff_mm:g} mm), bladder {first.bladder_cc:.1f} cc, "
                     f"V70/V60/V40 = {first.V70Gy_pct:.1f}/{first.V60Gy_pct:.1f}/{first.V40Gy_pct:.1f}%, "
                     f"exceeded: {first.limits_exceeded}")
    open(os.path.join(a.out, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # figure
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for d in dvhs[1:]:
        ax.plot(d.dose_Gy, 100 * d.volume_frac, color="0.75", lw=0.7)
    ax.plot(dvhs[0].dose_Gy, 100 * dvhs[0].volume_frac, color="k", lw=1.6, label="Planning contour")
    for row, sty in ((last, dict(color="k", ls="--", lw=1.4)), (first, dict(color="#A84B2F", lw=2.0))):
        if row is not None:
            d = dvhs[1 + int(S.index[S.level == row.level][0])]
            name = "Last level within limits" if row is last else "First level exceeding a limit"
            ax.plot(d.dose_Gy, 100 * d.volume_frac, label=f"{name} (d = {row.d_nominal_mm:g} mm)", **sty)
    for x, lim in LIMITS.items():
        ax.plot([x - 3, x + 3], [100 * lim] * 2, color="#20808D", lw=3)
    ax.set_xlabel("Dose (Gy)"); ax.set_ylabel("Bladder wall volume (%)")
    ax.set_xlim(0, 85); ax.set_ylim(0, 102); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "dvh.png"), dpi=200)
    if show:
        plt.show()
    plt.close(fig)

    # sagittal view of the deformation (planning contour, last level within limits, first exceedance)
    if first is not None:
        keys = ["planning contour"] + ([last.level] if last is not None else []) + [first.level]
    else:
        lv = list(S.level)
        keys = ["planning contour", lv[len(lv) // 2], lv[-1]]
    ct = load_ct_sagittal(find_ct(dicom_dir) if dicom_dir else [], C)
    fig = sagittal_figure(C, masks, keys, L, first, last, ct, plt)
    fig.savefig(os.path.join(a.out, "deformation.png"), dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    return L


# ------------------------------------------------------------------ sagittal figure
def find_ct(dicom_dir):
    """CT image files in dicom_dir (optional, used only as background of deformation.png)."""
    out = []
    for f in sorted(glob.glob(os.path.join(dicom_dir, "**", "*"), recursive=True)):
        if not os.path.isfile(f):
            continue
        try:
            h = pydicom.dcmread(f, stop_before_pixels=True, force=True)
        except Exception:
            continue
        if getattr(h, "Modality", "") == "CT":
            out.append((float(h.ImagePositionPatient[2]), f))
    return [f for _, f in sorted(out)]


def load_ct_sagittal(files, C):
    """Sagittal CT plane through the bladder centre (None if no CT is available)."""
    if not files:
        return None
    x_mm = C.ipp[0] + C.sag_i * C.dx
    cols, zs, ys = [], [], None
    for f in files:
        ds = pydicom.dcmread(f, force=True)
        img = ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1)) \
            + float(getattr(ds, "RescaleIntercept", 0))
        ipp = np.array(ds.ImagePositionPatient, float)
        ic = int(round((x_mm - ipp[0]) / float(ds.PixelSpacing[1])))
        if not 0 <= ic < img.shape[1]:
            return None
        cols.append(img[:, ic]); zs.append(ipp[2])
        if ys is None:
            ys = ipp[1] + np.arange(img.shape[0]) * float(ds.PixelSpacing[0])
    return np.array(cols), np.array(zs), ys


def sagittal_figure(C, masks, keys, L, first, last, ct, plt):
    """Planned dose with the planning bladder (dashed) and the simulated bladder (solid) on a sagittal plane."""
    i = C.sag_i
    y = C.ipp[1] + np.arange(C.Ny) * C.dy
    z = C.z
    b0 = C.bladder[:, :, i]
    roi = b0 | C.prostate[:, :, i]
    kk, jj = np.where(roi)
    ylim = (y[jj.min()] - 25, y[jj.max()] + 25)
    zlim = (z[kk.min()] - 25, z[kk.max()] + 25)
    ext = [y[0] - C.dy / 2, y[-1] + C.dy / 2, z[0] - C.dz / 2, z[-1] + C.dz / 2]
    dose = np.ma.masked_less(C.dose[:, :, i], 10.0)
    vmax = float(np.ceil(C.dose.max()))
    fig, axes = plt.subplots(1, len(keys), figsize=(3.0 * len(keys) + 0.8, 3.6), squeeze=False)
    for ax, key in zip(axes[0], keys):
        ax.set_facecolor("black")
        if ct is not None:
            img, cz, cy = ct
            ax.imshow(img, cmap="gray", vmin=-160, vmax=240, origin="lower", aspect="equal",
                      extent=[cy[0], cy[-1], cz[0], cz[-1]])
        im = ax.imshow(dose, cmap="turbo", vmin=10, vmax=vmax, alpha=0.45 if ct is not None else 1.0,
                       origin="lower", aspect="equal", extent=ext, interpolation="bilinear")
        ax.contour(y, z, b0.astype(float), levels=[0.5], colors="white", linewidths=1.0,
                   linestyles="--" if key != "planning contour" else "-")
        if key != "planning contour":
            ax.contour(y, z, masks[key][:, :, i].astype(float), levels=[0.5], colors="white", linewidths=1.6)
        ax.contour(y, z, C.prostate[:, :, i].astype(float), levels=[0.5], colors="black", linewidths=1.0,
                   linestyles=":")
        ax.set_xlim(ylim); ax.set_ylim(zlim); ax.set_xticks([]); ax.set_yticks([])
        r = L[L.level == key].iloc[0]
        tag = ("First level exceeding a limit" if first is not None and key == first.level else
               "Last level within limits" if last is not None and key == last.level else
               "Planning contour" if key == "planning contour" else "Simulated level")
        sub = "" if key == "planning contour" else f"d = {r.d_nominal_mm:g} mm (effective {r.dome_descent_eff_mm:g} mm), "
        ax.set_title(f"{tag}\n{sub}{r.bladder_cc:.0f} cc", fontsize=8, loc="left")
    a0 = axes[0][0]
    ant_left = C.col_dir[1] > 0          # +y is posterior in DICOM patient coordinates
    for x, t in ((0.03, "A" if ant_left else "P"), (0.97, "P" if ant_left else "A")):
        a0.text(x, 0.5, t, transform=a0.transAxes, color="white", fontsize=8, fontweight="bold",
                ha="left" if x < 0.5 else "right", va="center")
    a0.text(0.5, 0.97, "S", transform=a0.transAxes, color="white", fontsize=8, fontweight="bold", ha="center", va="top")
    fig.subplots_adjust(left=0.02, right=0.88, top=0.83, bottom=0.04, wspace=0.05)
    cb = fig.colorbar(im, cax=fig.add_axes([0.90, 0.12, 0.015, 0.66])); cb.set_label("Planned dose (Gy)")
    fig.text(0.02, 0.005, "Dashed white, planning bladder; solid white, simulated bladder; dotted black, prostate. "
             "Sagittal plane through the bladder centre.", fontsize=6.5)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dicom_dir", default=None, help="folder with the RTDOSE and RTSTRUCT of one patient")
    ap.add_argument("--rtdose", default=None)
    ap.add_argument("--rtstruct", default=None)
    ap.add_argument("--bladder", default="Bladder", help="ROI name of the planning bladder")
    ap.add_argument("--prostate", default="Prostate", help="ROI name of the prostate (CTV)")
    ap.add_argument("--step", type=float, default=STEP_MM, help="step of the nominal level (mm)")
    ap.add_argument("--out", default="results")
    a, _ = ap.parse_known_args()          # parse_known_args: ignores the extra arguments added by Colab
    if a.dicom_dir is None and (a.rtdose is None or a.rtstruct is None):
        a.dicom_dir = "/content" if os.path.isdir("/content") else "."
    run(a.dicom_dir, a.rtdose, a.rtstruct, a.bladder, a.prostate, a.step, a.out)


if __name__ == "__main__":
    main()
