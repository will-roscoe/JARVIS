#!/usr/bin/env python3
import datetime
import os

from astropy.io import fits
from jarvis import fpath
from jarvis.cvis import generate_coadded_fits, generate_rollings
from jarvis.extensions import extract_conf, pathfinder
from jarvis.power import powercalc
from jarvis.stats import correlate, stats  # noqa: F401
from jarvis.utils import await_confirmation, ensure_dir, hst_fpath_dict, hst_fpath_segdict, rpath, split_path, translate
from tqdm import tqdm

# makes an image
# n = fpath(r'datasets\HST\v04\jup_16-140-20-48-59_0103_v04_stis_f25srf2_proj.fits')
# fitsfile = fits.open(n)
# moind(fitsfile)
# makes a gif
# make_gif('datasets/HST/v04', dpi=300) #moonfp=True, remove_temp=False)
# hdu = fits.open(n)
# d = hdu[1].data
# print(d)
# NOTE: we use this condition to run the code only if it is run as a script to avoid running it when imported, this is best practice.
if (
    __name__ == "__main__"
):  # __name__ is a special,file-unique variable that is set to '__main__' when the script is run as a script, and set to the name of the module when imported.
    gps = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    # remove 3 (broken), 20 (southern) #group Numbe
    # loop to generate the coadded fits, identify the boundary, and calculate the power
    # fpaths = hst_fpath_segdict(2,False)
    # fsegs = hst_fpath_segdict(segments,kwargs.get("byvisit",False))
    #     fpaths = hst_fpath_dict(kwargs.get("byvisit",False))

    # ~ First generate coadds in each segment
    def gen_segmented_coadds(
        segpaths,
        coadd_dir=fpath("temp/coadds"),
        generate=True,
        kernel_params=(3, 1),
        overwrite=True,
        indiv=False,
        coadded=True,
        progressbar=True,
        confirm=False,
        **kwargs,
    ):
        if not (await_confirmation("Generate coadded fits in each segment.") if confirm else True):
            return None
        copaths = []
        ensure_dir(coadd_dir)
        if progressbar is True:
            pbar1 = tqdm(total=len(segpaths), desc="Generating coadded fits per segment")
        elif progressbar is not None:
            pbar1 = progressbar
            pbar1.reset(total=len(segpaths))
        for visit, fitpaths in segpaths.items():
            path = coadd_dir + "/" + visit + ".fits"
            if generate:
                fobj = generate_coadded_fits(
                    [fits.open(fp) for fp in fitpaths],
                    saveto=path,
                    kernel_params=kernel_params,
                    overwrite=overwrite,
                    indiv=indiv,
                    coadded=coadded,
                    **kwargs,
                )
                fobj.close()
            copaths.append([visit, path])
            if pbar1 is not None:
                pbar1.update()
                pbar1.set_postfix_str(f"{visit}")
        if progressbar is True:
            pbar1.close()
        return copaths

    # #! then run pathfinder on each segment.
    def segmented_pathfinder(copaths, extname="BOUNDARY", progressbar=True, confirm=False, **kwargs):
        """copaths: dict containing the visit:path to the coadded fits"""
        if not (await_confirmation("Run pathfinder on each segment.") if confirm else True):
            return None
        if progressbar is True:
            pbar2 = tqdm(total=len(copaths), desc="Running pathfinder on each segment")
        elif progressbar is not None:
            pbar2 = progressbar
            pbar2.reset(total=len(copaths))
        for i, [visit, copath] in enumerate(copaths):
            ret = pathfinder(copath, steps=False, extname=extname, **kwargs)
            if ret[0]:
                if progressbar is not None:
                    pbar2.set_postfix_str(f"Pathfinder succeeded on {copath=}")
            else:
                if progressbar is not None:
                    pbar2.set_postfix_str(f"Pathfinder failed on {copath=}")
                copaths.pop(i)
            if progressbar is not None:
                pbar2.update()
                pbar2.set_postfix_str(f"{visit}")
        if progressbar is True:
            pbar2.close()
        return copaths

    # #! generate rolling averages of the fits, in their respective groups, and then save and split them into their respective segments
    def gen_averages(
        fpaths,
        segpaths,
        avg_dir=fpath("temp/rollavgs"),
        generate=True,
        window=3,
        kernel_params=(3, 1),
        indiv=True,
        coadded=True,
        overwrite=True,
        progressbar=True,
        confirm=False,
        **kwargs,
    ):
        if not (await_confirmation("Generate denoised individual fits") if confirm else True):
            return None
        if progressbar is True:
            pbar3 = tqdm(total=sum([len(fpaths[i]) for i in fpaths]), desc="Generating denoised fits")
        elif progressbar is not None:
            pbar3 = progressbar
            pbar3.reset(total=sum([len(fpaths[i]) for i in fpaths]))
        avgpaths = {}
        for k, f in fpaths.items():
            # from fsegs identify the "visit" key that contains the fits obj
            # then turn into the correct dir
            if generate:
                fitsobjs = generate_rollings(
                    [fits.open(ff) for ff in f],
                    window=window,
                    kernel_params=kernel_params,
                    indiv=indiv,
                    coadded=coadded,
                    **kwargs,
                )
            for i in range(len(f)):
                if generate:
                    fobj = fitsobjs[i]
                initpath = f[i]
                visit = None
                for key, value in segpaths.items():
                    if initpath in value:
                        visit = key
                        break
                avgpath = avg_dir + f"/{visit}/" + initpath.split("/")[-1].split("\\")[-1]
                if visit in avgpaths:
                    avgpaths[visit] = avgpaths[visit] + [avgpath]
                else:
                    avgpaths[visit] = [avgpath]
                    ensure_dir(avg_dir + f"/{visit}")
                if generate:
                    fobj.writeto(avgpath, overwrite=overwrite)
                    fobj.close()
                if progressbar is not None:
                    pbar3.update()
                    pbar3.set_postfix_str(f"{i+1}/{len(f)} in {visit}")
        if progressbar is True:
            pbar3.close()
        return avgpaths

    # #! using each pathed coadded fit, run pathfinder with conf, in silent mode, and then run powercalc.
    def gen_path_powercalc(
        avgpaths,
        coaddir=fpath("temp/coadds"),
        outfile="auto",
        extname="BOUNDARY",
        progressbar=True,
        confirm=False,
        overwrite=True,
        force=False,
        **kwargs,
    ):
        failed = []
        ignores, conf_fpath, conf_as_kwargs = (
            kwargs.pop("ignores", []),
            kwargs.pop("conf_fpath", None),
            kwargs.pop("conf_as_kwargs", False),
        )
        if not (await_confirmation("Generate power fluxes.") if confirm else True):
            return
        if progressbar is True:
            pbar4 = tqdm(total=sum(len(avgpaths[i]) for i in avgpaths), desc="Generating Power Fluxes")
        elif progressbar is not None:
            pbar4 = progressbar
            pbar4.reset(total=sum(len(avgpaths[i]) for i in avgpaths))
        outfile = (
            f"{extname.upper()}_" + datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S") + ".txt"
            if outfile == "auto"
            else outfile
        )
        if overwrite:
            open(outfile, "w").close()
        else:
            open(outfile, "a").close()
        if conf_fpath is not None:
            alt_conf = extract_conf(fits.getheader(conf_fpath, extname, kwargs.get("extver")), ignore=ignores)
        for visit, fitspaths in avgpaths.items():
            copath = conf_fpath if conf_fpath else coaddir + "/" + visit + ".fits"
            if not conf_fpath:  # if alternative conf is not provided
                try:  # try to get the header from the coadded fits at copath and extract the conf
                    conf_header = fits.getheader(copath, extname, kwargs.get("extver"))
                    conf = extract_conf(conf_header, ignore=ignores)
                except Exception as e:
                    tqdm.write(
                        f"Error: {e}, Failed to get header from {copath=} with {extname=}, {kwargs.get("extver")=}",
                    )
                    conf = None
            else:
                conf = alt_conf
            if force or conf:
                for j, f in enumerate(fitspaths):
                    persists = {"extname": extname, "steps": False, "show": False}
                    if conf_as_kwargs:
                        persists.update(conf)
                    else:
                        persists["conf"] = conf
                    persists.update({})
                    persists.update(kwargs)
                    pf = pathfinder(f, steps=False, show=False, **persists)
                    path = pf[1]
                    if pf[0]:
                        pc = powercalc(fits.open(f), path, writeto=outfile, extname=extname)
                        if progressbar is not None:
                            pbar4.update()
                            pbar4.set_postfix_str(f"P={pc[0]:.3f}GW,I={pc[1]*1e8:.3f}x10⁻⁸GW/km²")
                    else:
                        failed.append([f, copath, path])  # this is a list of paths that failed pathfinder.
                        if progressbar is not None:
                            pbar4.update()
                            pbar4.set_postfix_str(f"Failed on {"/".join(split_path(f)[-2:])}")
                        tqdm.write(f"F{len(failed):0>3}failed on: {rpath(f)} using config: {path}")
            else:
                if progressbar is not None:
                    pbar4.update(len(fitspaths))
                tqdm.write(f"No boundary found for {visit=!s}")
        if progressbar is True:
            pbar4.close()
        for file in failed:
            print(f"failed on: {rpath(file[0])} using {rpath(file[1])} config: {file[2]}")
        print(f"{len(failed)=}")
        print(f"Power fluxes written to {outfile}")

    def run_path_powercalc(
        ignores={"groups": [], "visits": []},
        byvisit=True,
        segments=3,
        coadd_dir=fpath("temp/coadds"),
        avg_dir=fpath("temp/rollavgs"),
        outfile=fpath("temp/powers.txt"),
        extname="BOUNDARY",
        remove="none",
    ):
        filepaths_byv = hst_fpath_dict(byvisit=byvisit)
        filepaths_seg = hst_fpath_segdict(segments, byvisit=byvisit)

        # to ignore any visit or group (or segment in either), we translate all to the correct form and format.
        ignores_ = []
        ident = "v" if byvisit else "g"
        ign = [item for sublist in [[[k, v] for v in ignores[k]] for k in ignores] for item in sublist]
        for iden, num in ign:
            if isinstance(num, int):
                ignores_.append(f"{ident}{translate(num, init=iden,to=ident):0>2}")
            else:
                ignores_.append(f"{ident}{num[0] if num[0]!=iden[0] else ''}{num[1:]}")
        for visit in filepaths_byv:
            if any(i in visit for i in ignores_):
                filepaths_byv.pop(visit)
        for visit in filepaths_seg:
            if any(i in visit for i in ignores_):
                filepaths_seg.pop(visit)
        ## main part of function

        # generate coadded fits in each segment
        gen = await_confirmation("regenerate coadded fits?")
        copaths = gen_segmented_coadds(filepaths_seg, coadd_dir, generate=gen, confirm=False)
        # run pathfinder on each segment
        if await_confirmation("find paths on each coadd (per segment)?"):
            copaths = segmented_pathfinder(copaths, extname, confirm=False)
        # generate rolling averages of the fits, in their respective groups, and then save and split them into their respective segments
        gen = await_confirmation("regenerate rolling averages?")
        avgpaths = gen_averages(filepaths_byv, filepaths_seg, avg_dir, generate=gen, confirm=False)
        # using each pathed coadded fit, run pathfinder with conf, in silent mode, and then run powercalc.
        gen_path_powercalc(avgpaths, coadd_dir, outfile, extname, confirm=False, overwrite=True, force=False)
        if remove in ["both", "coadds"]:
            for visit in filepaths_seg:
                os.remove(coadd_dir + f"/{visit}.fits")
        if remove in ["both", "rollavgs"]:
            for visit in avgpaths:
                for p in os.listdir(avg_dir + f"/{visit}"):
                    if os.path.isfile(p):
                        os.remove(p)

    run_path_powercalc()

    # except KeyError:
    #    fit.close()
    #    print(f'No boundary found for {visit=}')
# loop to only generate the gaussians ----------------------------------------
#     for i in tqdm(gps):
#         basefitpath = fpath(f'datasets/HST/group_{i:0>2}')
#         fitsg = fits_from_glob(basefitpath)
#         copath = fpath(f'datasets/HST/custom/g{i:0>2},v{group_to_visit(i):0>2}_[3,1]gaussian-coadded.fits')
#         fit = generate_coadded_fits(fitsg, saveto=copath, gaussian=(3,1), overwrite=True,indiv=False, coadded=True)
#         fit.close()
# # loop to only run pathfinder ------------------------------------------------
# for i in tqdm(gps):
#     copath = fpath(f'datasets/HST/custom/g{i:0>2},v{group_to_visit(i):0>2}_[3,1]gaussian-coadded.fits')
#     pt = pathfinder(copath)
# loop to only run powercalc -------------------------------------------------
# for i in tqdm(gps):
#     basefitpath = fpath(f'datasets/HST/group_{i:0>2}')
#     fitsg = fits_from_glob(basefitpath)
#     copath = fpath(f'datasets/HST/custom/g{i:0>2},v{group_to_visit(i):0>2}_[3,1]gaussian-coadded.fits')
#     fit = fits.open(copath)
#     try:
#         path = np.array(fit['BOUNDARY'].data.tolist())
#         fit.close()
#         pbr = tqdm(total=len(fitsg), desc=f'"powercalc"(group {i} of {len(gps)})')
#         for j,f in enumerate(fitsg):
#             pc = powercalc(f,path, writeto=outfile)
#             pbr.set_postfix_str(f"P={pc[0]:.3f}GW,I={pc[1]*1e8:.3f}x10⁻⁸GW/km²")
#             pbr.update()
#             f.close()
#         pbr.close()
#     except KeyError:
#         fit.close()
#         print(f'No boundary found for group {i}')
## loop to generate gaussians of segments of the fits

# pbar = tqdm(total=len(fpaths), desc='Generating coadded fits')
# for k,f in fpaths.items():
#     fitsobjs = generate_rollings([fits.open(ff) for ff in f], kernel_params=(3,1),indiv=True, coadded=True)
#     ensure_dir(savedir+f'/{k}')
#     for i,ff in enumerate(fitsobjs):
#         ff.writeto(fpath(savedir+f'/{k}/{get_datetime(ff).strftime("%Y_%m_%dT%H_%M_%S")}window3_.fits'), overwrite=True)
#     pbar.update()
# pbar.close()
## loop to run pathfinder on segments of the fits
# lrange = (0.25,0.32)

# for g in fpaths.keys():
#     #if int(g[1:3]) < 7 or if g=='g07sA':
#         copath =fpath(f'datasets/HST/custom/{g}_[3,1]gaussian-coadded.fits')
#         if os.path.isfile(copath):
#             pt = pathfinder(copath, fixlrange=lrange)
## loop to run powercalc on segments of the fits
# for g,fs in fpaths.items():
#     copath = fpath(f'datasets/HST/custom/{g}_[3,1]gaussian-coadded.fits')
#     if os.path.isfile(copath):
#         fit = fits.open(copath)
#         try:
#             cpath = np.array(fit['BOUNDARY'].data.tolist())

#             pbr = tqdm(total=len(fs), desc=f'"powercalc"({g})')
#             for f in fs:
#                 ff = fits.open(f)
#                 pc = powercalc(ff, cpath, writeto=outfile)
#                 pbr.set_postfix_str(f"P={pc[0]:.3f}GW,I={pc[1]*1e8:.3f}x10⁻⁸GW/km²")
#                 pbr.update()
#                 ff.close()
#             pbr.close()
#         except KeyError:
#             fit.close()
#             print(f'No boundary found for {g}')


# statsfile = 'powers.txt'
# table = pd.read_csv(statsfile, header=0,sep=r'\s+')
# print(table.columns)
# print(table)
# print(table)
# print(stats(table['PFlux'],mean=True,median=True,std=True))

# statsfile_2 = (fpath(r'datasets\Solar_Wind.txt'))
# table_2 = pd.read_csv(statsfile_2, header=0,sep=r'\s+')
# table_2 = table_2.iloc[308:310]
# print(table_2)
# correlated = correlate(table['PFlux'], table_2['jup_sw_pdyn'])

# table.plot(x='Time', y='PFlux', title='Power Flux vs Time at Jupiter', xlabel='Time [h:m:s]', ylabel='PFlux [W/m^2]')
# plt.title("Variation in power flux")
# plt.xlabel("Time [h:m:s]")
# plt.ylabel("Power Flux [W/m^2]")
# plt.show()


filepath = fpath("data/...")
