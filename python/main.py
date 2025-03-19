#!/usr/bin/env python3
from contextlib import suppress
import datetime
from glob import glob
import os
import re
import shutil

from astropy.io import fits
from matplotlib import figure, pyplot as plt
import numpy as np
from jarvis import fpath, fits_from_glob
from jarvis.cvis import generate_coadded_fits, generate_rollings
from jarvis.extensions import extract_conf, pathfinder, power_gif
from jarvis.power import powercalc
#from jarvis.stats import correlate, stats  # noqa: F401
from jarvis.utils import await_confirmation, dump_powerdicts, ensure_dir, ensure_file, fitsheader, get_datapaths, get_time_interval_from_multi, hisaki_sw_get_safe, hst_fpath_dict, hst_fpath_segdict, merge_fdicts, prepdfs, rpath, split_path, statusprint, translate, jprofile
from tqdm import tqdm
from jarvis.const import HISAKI, HST, Dirs, log
from jarvis.plotting import apply_plot_defaults, dpr_histogram_plot, figsize, megafigure_plot, overlaid_plot, stacked_plot
import cutie

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
    @jprofile("generate_coadd_segmented")
    def gen_segmented_coadds(
        segpaths,
        coadd_dir=fpath("temp/coadds"),
        generate=True,
        kernel_params=(3, 1),
        overwrite=True,
        indiv=False,
        coadded=True,
        progressbar=True,
        **kwargs,
    ):
        """Generate coadded fits files for each visit, per segment.

        Args:
            segpaths (dict): dict containing the visit:path to the fits files
            coadd_dir (str, optional): directory to save the coadded fits. Defaults to fpath("temp/coadds").
            generate (bool, optional): whether to generate the coadded fits. Defaults to True.
            kernel_params (tuple, optional): kernel parameters for the coaddition. Defaults to (3, 1).
            overwrite (bool, optional): whether to overwrite existing files. Defaults to True.
            indiv (bool, optional): whether to save the individual fits. Defaults to False.
            coadded (bool, optional): whether to save the coadded fits. Defaults to True.
            progressbar (bool, optional): whether to show a progressbar. Defaults to True.
            confirm (bool, optional): whether to confirm before running. Defaults to False.

        Returns:
            list: list of lists containing the visit and path to the coadded fits. in the form [[visit, path],...]
        
        """
        copaths = []
        ensure_dir(coadd_dir)
        if progressbar is True:
            pbar1 = tqdm(total=len(segpaths), desc="Generating coadded fits per segment")
        elif progressbar is not None:
            pbar1 = progressbar
            pbar1.reset(total=len(segpaths))
        else:
            pbar1 = None
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
    @jprofile("segmented_pathfinder")
    def segmented_pathfinder(copaths, extname="BOUNDARY", progressbar=True, **kwargs):
        """Run pathfinder on each segment (Generates paths for each coadded fit).

        Args:
            copaths (list): list of lists containing the visit and path to the coadded fits. in the form [[visit, path],...]
            extname (str, optional): extname of the boundary data. Defaults to "BOUNDARY".
            progressbar (bool, optional): whether to show a progressbar. Defaults to True.
            confirm (bool, optional): whether to confirm before running. Defaults to False.
            **kwargs: additional keyword arguments to pass to pathfinder.
                - ignores (list, optional): list of keys to ignore in the conf. Defaults to [].
                - conf_fpath (str, optional): path to the conf fits file. Defaults to None.
                - conf_as_kwargs (bool, optional): whether to pass the conf as
                    kwargs instead of conf. Defaults to False.

        Returns:
            list: list of lists containing the visit and path to the coadded fits. in the form [[visit, path],...]

        """
        if progressbar is True:
            pbar2 = tqdm(total=len(copaths), desc="Running pathfinder on each segment")
        elif progressbar is not None:
            pbar2 = progressbar
            pbar2.reset(total=len(copaths))
        else:
            pbar2 = None
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


    @jprofile("visit_powercalc")
    def visit_powercalc(copaths, extname="BOUNDARY", progressbar=True, outfile="auto", overwrite=True):
        """Generate power fluxes from coadded fits.
        
        Args:
            copaths (list): list of lists containing the visit and path to the coadded fits. in the form [[visit, path],...]
            extname (str, optional): extname of the boundary data. Defaults to "BOUNDARY".
            progressbar (bool, optional): whether to show a progressbar. Defaults to True.
            outfile (str, optional): path to the output file. Defaults to "auto".
            overwrite (bool, optional): whether to overwrite the output file. Defaults to True.
            force (bool, optional): whether to force the powercalc. Defaults to False.

        Returns:
            list: list of dicts containing the visit, power, flux, and area. in the form [{"visit": visit, "power": power, "flux": flux, "area": area

        """
        fstream = ""
        fdicts = []
        failed = []
        if progressbar is True:
            pbar4 = tqdm(total=len(copaths), desc="Generating Power Fluxes")
        elif progressbar is not None:
            pbar4 = progressbar
            pbar4.reset(total=len(copaths))
        else:
            pbar4 = None
        outfile = (
            str(Dirs.GEN)+f"/{extname.upper()}_coadds_" + datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S") + ".txt"
            if outfile == "auto"
            else outfile
        )
        ensure_file(outfile)
        if overwrite:
            open(outfile, "w").close()
        else:
            open(outfile, "a").close()
        for j, (visit,f) in enumerate(copaths):
            copath = fits.open(f)
            if extname in copath:
                path = np.array(copath[extname].data.tolist())
                copath.close()
                pc = powercalc(fits.open(f), path, writeto=outfile)
                fstream+= f'{pc["visit"]} {pc["power"]} {pc["flux"]} {pc["area"]}\n'
                fdicts.append(pc)
                if progressbar is not None:
                    pbar4.update()
                    pbar4.set_postfix_str(f"SR:{statusprint(len(fdicts)/(len(fdicts)+len(failed)), True)}, P={pc["power"]:.3f}GW,I={pc["flux"]*1e8:.3f}x10⁻⁸GW/km²")
            else:
                failed.append([f, copath, path])  # this is a list of paths that failed pathfinder.
                if progressbar is not None:
                    pbar4.update()
                    pbar4.set_postfix_str(f"FR:{statusprint(len(failed)/(len(failed)+len(fdicts)))}, Failed on {"".join(split_path(f)[-2:])[:40]+"..."},")
                log.write(f"F{len(failed):0>3} failed on: {rpath(f)} using config: {path}")
        if progressbar is True:
            pbar4.close()
        log.write(f"Succcesful Calcs:\n {fstream}")
        tqdm.write(f"{len(failed)=}/{len(failed)+ len(fdicts)}")
        tqdm.write(f"Power fluxes written to {outfile}")
        tqdm.write(fstream)
        with open(outfile) as f:
            if not f.readlines():
                os.remove(outfile)
        return fdicts


    # #! generate rolling averages of the fits, in their respective groups, and then save and split them into their respective segments
    @jprofile("generate_averages")
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
        **kwargs,
    ):
        """Generate rolling averages of the fits, in their respective groups, and then save and split them into their respective segments.

        Args:
            fpaths (dict): dict containing the visit:path to the fits files
            segpaths (dict): dict containing the visit:path to the fits files
            avg_dir (str, optional): directory to save the rolling averages. Defaults to fpath("temp/rollavgs").
            generate (bool, optional): whether to generate the rolling averages. Defaults to True.
            window (int, optional): window size for the rolling average. Defaults to 3.
            kernel_params (tuple, optional): kernel parameters for the rolling average. Defaults to (3, 1).
            indiv (bool, optional): whether to save the individual fits. Defaults to True.
            coadded (bool, optional): whether to save the coadded fits. Defaults to True.
            overwrite (bool, optional): whether to overwrite existing files. Defaults to True.
            progressbar (bool, optional): whether to show a progressbar. Defaults to True.
            confirm (bool, optional): whether to confirm before running. Defaults to False.
            **kwargs: additional keyword arguments to pass to generate_rollings.

        Returns:
            dict: dict containing the visit:path to the rolling averages. in the form {visit: [path, ...]
        
        """
        if progressbar is True:
            pbar3 = tqdm(total=sum([len(fpaths[i]) for i in fpaths]), desc="Generating denoised fits")
        elif progressbar is not None:
            pbar3 = progressbar
            pbar3.reset(total=sum([len(fpaths[i]) for i in fpaths]))
        else:
            pbar3 = None
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
    @jprofile("gen_path_powercalc")
    def gen_path_powercalc(
        avgpaths,
        coaddir=fpath("temp/coadds"),
        outfile="auto",
        extname="BOUNDARY",
        progressbar=True,
        overwrite=True,
        force=False,
        **kwargs,
    ):
        """Generate power fluxes from coadded fits.

        Args:
            avgpaths (dict): dict containing the visit:path to the fits files
            coaddir (str, optional): directory containing the coadded fits. Defaults to fpath("temp/coadds").
            outfile (str, optional): path to the output file. Defaults to "auto".
            extname (str, optional): extname of the boundary data. Defaults to "BOUNDARY".
            progressbar (bool, optional): whether to show a progressbar. Defaults to True.
            confirm (bool, optional): whether to confirm before running. Defaults to False.
            overwrite (bool, optional): whether to overwrite the output file. Defaults to True.
            force (bool, optional): whether to force the powercalc. Defaults to False.
            **kwargs: additional keyword arguments to pass to pathfinder.
                - ignores (list, optional): list of keys to ignore in the conf. Defaults to [].
                - conf_fpath (str, optional): path to the conf fits file. Defaults to None.
                - conf_as_kwargs (bool, optional): whether to pass the conf as
        Returns:
            list: list of dicts containing the visit, power, flux, and area. in the form [{"visit": visit, "power": power, "flux": flux, "area": area

        """
        fstream = ""
        fdicts = []
        failed = []
        ignores, conf_fpath, conf_as_kwargs = (
            kwargs.pop("ignores", []),
            kwargs.pop("conf_fpath", None),
            kwargs.pop("conf_as_kwargs", False),
        )
        if progressbar is True:
            pbar4 = tqdm(total=sum(len(avgpaths[i]) for i in avgpaths), desc="Generating Power Fluxes")
        elif progressbar is not None:
            pbar4 = progressbar
            pbar4.reset(total=sum(len(avgpaths[i]) for i in avgpaths))
        else:
            pbar4 = None
        outfile = (
            str(Dirs.GEN)+f"/{extname.upper()}_" + datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S") + ".txt"
            if outfile == "auto"
            else outfile
        )
        ensure_file(outfile)
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
                    persists = {"extname": extname, "steps": False, "show": False} #"old":True}
                    if conf_as_kwargs:
                        persists.update(conf)
                    else:
                        persists["conf"] = conf
                    persists.update({})
                    persists.update(kwargs)
                    pf = pathfinder(f,  **persists)
                    path = pf[1]
                    if pf[0]:
                        pc = powercalc(fits.open(f), path, writeto=outfile)
                        fstream+= f'{pc["visit"]} {pc["power"]} {pc["flux"]} {pc["area"]}\n'
                        fdicts.append(pc)
                        if progressbar is not None:
                            pbar4.update()
                            pbar4.set_postfix_str(f"SR:{statusprint(len(fdicts)/(len(fdicts)+len(failed)), True)}, P={pc["power"]:.3f}GW,I={pc["flux"]*1e8:.3f}x10⁻⁸GW/km²")
                    else:
                        failed.append([f, copath, path])  # this is a list of paths that failed pathfinder.
                        if progressbar is not None:
                            pbar4.update()
                            pbar4.set_postfix_str(f"FR:{statusprint(len(failed)/(len(failed)+len(fdicts)))}, Failed on {"".join(split_path(f)[-2:])[:40]+"..."},")
                        log.write(f"F{len(failed):0>3} failed on: {rpath(f)} using config: {path}")
            else:
                if progressbar is not None:
                    pbar4.update(len(fitspaths))
                tqdm.write(f"No boundary found for {visit=!s}")
        if progressbar is True:
            pbar4.close()
        log.write(f"Succcesful Calcs:\n {fstream}")
        tqdm.write(f"{len(failed)=}/{len(failed)+ len(fdicts)}")
        tqdm.write(f"Power fluxes written to {outfile}")
        tqdm.write(fstream)
        with open(outfile) as f:
            if not f.readlines():
                os.remove(outfile)
        return fdicts
    
    @jprofile("copath_avg_powercalc")
    def copath_avg_powercalc(
        avgpaths,
        coaddir=fpath("temp/coadds"),
        outfile="auto",
        extname="BOUNDARY",
        progressbar=True,
        overwrite=True,
        force=False,
        **kwargs,
    ):
        """Generate power fluxes from coadded fits.

        Args:
            avgpaths (dict): dict containing the visit:path to the fits files
            coaddir (str, optional): directory containing the coadded fits. Defaults to fpath("temp/coadds").
            outfile (str, optional): path to the output file. Defaults to "auto".
            extname (str, optional): extname of the boundary data. Defaults to "BOUNDARY".
            progressbar (bool, optional): whether to show a progressbar. Defaults to True.
            confirm (bool, optional): whether to confirm before running. Defaults to False.
            overwrite (bool, optional): whether to overwrite the output file. Defaults to True.
            force (bool, optional): whether to force the powercalc. Defaults to False.
            **kwargs: additional keyword arguments to pass to pathfinder.
                - ignores (list, optional): list of keys to ignore in the conf. Defaults to [].
                - conf_fpath (str, optional): path to the conf fits file. Defaults to None.
                - conf_as_kwargs (bool, optional): whether to pass the conf as
        Returns:
            list: list of dicts containing the visit, power, flux, and area. in the form [{"visit": visit, "power": power, "flux": flux, "area": area
        """
        fstream = ""
        fdicts = []
        failed = []
        ignores =kwargs.pop("ignores", [])
        if progressbar is True:
            pbar4 = tqdm(total=sum(len(avgpaths[i]) for i in avgpaths), desc="Generating Power Fluxes")
        elif progressbar is not None:
            pbar4 = progressbar
            pbar4.reset(total=sum(len(avgpaths[i]) for i in avgpaths))
        else: 
            pbar4 = None
        outfile = (
            str(Dirs.GEN)+f"/{extname.upper()}_copath_" + datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S") + ".txt"
            if outfile == "auto"
            else outfile
        )
        ensure_file(outfile)
        if overwrite:
            open(outfile, "w").close()
        else:
            open(outfile, "a").close()
        for visit, fitspaths in avgpaths.items():
            copath =  coaddir + "/" + visit + ".fits"
            try:  # try to get the header from the coadded fits at copath and extract the conf
                conf_path = fits.getdata(copath, extname, kwargs.get("extver"))
                fp = fits.open(copath)
                exinfo = {k:fitsheader(fp, k, ind=extname) for k in ["LMIN","LMAX","NUMPTS"]}
                fp.close()
            except Exception as e:
                tqdm.write(
                    f"Error: {e}, Failed to get header from {copath=} with {extname=}, {kwargs.get("extver")=}",
                )
                conf_path = None
            if force or conf_path is not None:
                for j, f in enumerate(fitspaths):
                    ft = fits.open(f)
                    pc = powercalc(ft, conf_path, writeto=outfile, exinfo=exinfo)
                    with suppress(Exception):
                        ft.close()
                    fstream+= f'{pc["visit"]} {pc["power"]} {pc["flux"]} {pc["area"]}\n'
                    fdicts.append(pc)
                    if progressbar is not None:
                        pbar4.update()
                        pbar4.set_postfix_str(f"SR:{statusprint(len(fdicts)/(len(fdicts)+len(failed)), True)}, P={pc["power"]:.3f}GW,I={pc["flux"]*1e8:.3f}x10⁻⁸GW/km²")     
            else:
                if progressbar is not None:
                    pbar4.update(len(fitspaths))
                tqdm.write(f"No boundary found for {visit=!s}")
        if progressbar is True:
            pbar4.close()
        log.write(f"Succcesful Calcs:\n {fstream}")
        # tqdm.write(f"{len(failed)=}/{len(failed)+ len(fdicts)}")
        # tqdm.write(f"Power fluxes written to {outfile}")
        # tqdm.write(fstream)
        # check if file actually has any lines in it and remove if not.
        with open(outfile) as f:
            if not f.readlines():
                os.remove(outfile)

        return fdicts
    @jprofile("figure_gen")
    def figure_gen(hst_cols,hisaki_cols,include="last5",stacked=False,overlaid=True,megafigure=True):
        # hisaki_colnames = list(HISAKI.colnames.values())
        # hst_colnames = list(HST.colnames.values())    
        # hst_cols = [h for h in hst_cols if h in hst_colnames]
        # hisaki_cols = [h for h in hisaki_cols if h in hisaki_colnames]
        overlaid_map = [
            *hst_cols,
            [["Torus_Power_Dawn","Torus_Power_Dusk"],"Aurora_Power"], # shared axis
            "Pdyn",
            "Aurora_Flux", # twin axis
        ]
        if "last" in include:
            num = int("".join([char if char.isnumeric() else "" for char in include])) if len(include)>4 else 1
            hst_datasets = get_datapaths()[0:num]
        else:
            hst_datasets = get_datapaths()
        hst_datasets = prepdfs(hst_datasets)
        hisaki_dataset = hisaki_sw_get_safe(method="all")
        time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        paths = []
        if stacked:
            paths.append(stacked_plot(hst_datasets=hst_datasets, hisaki_dataset=hisaki_dataset, hst_cols=hst_cols, hisaki_cols=hisaki_cols, savepath=fpath("figures/imgs/")))
            
        if overlaid: # plot all datasets on the same plot
            paths.append(overlaid_plot(overlaid_map, hst_datasets, hisaki_dataset, fpath("figures/imgs/")))
            
        if megafigure:
            paths.append(megafigure_plot(hst_datasets=hst_datasets, hisaki_dataset=hisaki_dataset, hst_cols=hst_cols, hisaki_cols=hisaki_cols, savepath=fpath("figures/imgs/")))
            # plot all datasets over full interval like overlaid, but then plot each visit separately below it in a grid.
            # example if we find 8 unique visits in the hst dataset, we will have a grid of 8 plots below the overlaid plot.
        return paths
    #@jprofile("main")
    def run_path_powercalc(
        ignores={"groups": [], "visits": ["v01","v02","v03","v04","v05","v06","v07","v08"]},
        byvisit=True,
        segments=2,
        coadd_dir=fpath("temp/coadds"),
        avg_dir=fpath("temp/rollavgs"),
        gifdir=fpath("figures/gifs"),
        outfile="auto",
        extname="BOUNDARY",
        remove="none",
        window=3,
        config = {},
        fig_config = {}
    ):
        global init_time
        paths = []
        init_time = datetime.datetime.now()
        def writelog(msg, ):
            timedelta = datetime.datetime.now() - init_time
            sec = timedelta.total_seconds()
            mins = int(sec//60)
            sec = sec%60
            msec = int(sec%1 * 1000)
            sec = int(sec)
            tqdm.write(f"[{mins:0>3}:{sec:0>2}:{msec:0>4}] {msg}")

        writelog("Starting...")
        filepaths_byv = hst_fpath_dict(byvisit=byvisit)
        filepaths_seg = hst_fpath_segdict(segments, byvisit=byvisit)
        # config is used to predefine the anwsers to the confirmation prompts.
        # config = {"generate_coadd":bool, "coadd_pathfinder":bool, "coadd_power":bool, "generate_avgs":bool, "avg_power":bool, "avg_pathfinder":bool, "gif":bool}
        # to ignore any visit or group (or segment in either), we translate all to the correct form and format.
        
        ignores_ = ["v05","v29"]
        ident = "v" if byvisit else "g"
        ign = [item for sublist in [[[k, v] for v in ignores[k]] for k in ignores] for item in sublist]
        for iden, num in ign:
            if isinstance(num, int):
                l = translate(num, init=iden,to=ident)
                ignores_.append(f"{ident}{l:0>2}")
                for i in range(num):
                    tqdm.write(f"{iden}{num[i]}-->{l[i]}")
            else:
                ignores_.append(f"{ident}{num[0] if num[0]!=iden[0] else ''}{num[1:]}")
        destbyv = []
        for visit in filepaths_byv:
            if any(i in visit for i in ignores_):
                destbyv.append(visit)
        destseg = []
        for visit in filepaths_seg:
            if any(i in visit for i in ignores_):
                destseg.append(visit)
        for i in destbyv:
            filepaths_byv.pop(i)
        for i in destseg:
            filepaths_seg.pop(i)
        writelog("Visits Selected = "+",".join(filepaths_byv.keys()))
        writelog("Organized Paths")
        ### MAIN PART #########################################################
        #---- OPT1: Generate coadded fits for each segment
        gen = await_confirmation("regenerate coadded fits?") if "generate_coadd" not in config else config["generate_coadd"]
        if gen:
            writelog("Generating Coadds...")
            writelog(",".join(list(filepaths_seg.keys())))
        copaths = gen_segmented_coadds(filepaths_seg, coadd_dir, generate=gen, progressbar=True if gen else None)
        
        if gen:
            writelog("Coadds Generated")
        #---- OPT2: run pathfinder on each segment
        if await_confirmation("find paths on each coadd (per segment)?") if "coadd_pathfinder" not in config else config["coadd_pathfinder"]:
            writelog("Running pathfinder on each segment...")
            copaths = segmented_pathfinder(copaths, extname)
            
            writelog("Pathfinder Complete")
        #---- OPT3: generate power fluxes from coadded fits (from OPT1->2)
        if await_confirmation("generate approximate power fluxes from coadds?") if "coadd_power" not in config else config["coadd_power"]:
            writelog("Generating Power Fluxes...")
            coadd_gen = visit_powercalc(copaths, extname, outfile=outfile, overwrite=True,)
            writelog("Power Fluxes Generated")
        #---- OPT4: generate rolling averages of the orig. fits files
        # generate rolling averages of the fits, in their respective groups, and then save and split them into their respective segments
        gen= await_confirmation("regenerate rolling averages?") if "generate_avgs" not in config else config["generate_avgs"]
        # we have to always run this to get the paths out.
        if gen:
            writelog("Generating Rolling Averages...")
        avgpaths = gen_averages(filepaths_byv, filepaths_seg, avg_dir, generate=gen, window=window, progressbar=True if gen else None)
        if gen:
            writelog("Rolling Averages Generated")
        # using each pathed coadded fit, run pathfinder with conf, in silent mode, and then run powercalc.
        #---- OPT5: generate power fluxes on each rolling average (from OPT4)
        if await_confirmation("generate power fluxes on each rolling average using paths?") if "avg_power" not in config else config["avg_power"]:
            #---- 0PT5A: generate paths on each rolling average, or use the paths from the coadds?
            if await_confirmation("Generate paths on each rolling average? (Y) or use the paths from the coadds? (N)") if "avg_pathfinder" not in config else config["avg_pathfinder"]:
                writelog("Running pathfinder on each rolling average, and then powercalc...")
                fdicts = gen_path_powercalc(avgpaths, coadd_dir, outfile, extname,  overwrite=True, force=False)
            else:
                writelog("Running powercalc on each rolling average...")
                fdicts = copath_avg_powercalc(avgpaths, coadd_dir, outfile, extname,  overwrite=True, force=False)
            writelog("Power Fluxes Generated")
        writelog("Cleaning up generated files...")
        nu = 0
        if remove in ["all", "coadds"]:
            for visit in filepaths_seg:
                os.remove(coadd_dir + f"/{visit}.fits")
                nu+=1
        if remove in ["all", "rollavgs"]:
            for visit in avgpaths:
                for p in os.listdir(avg_dir + f"/{visit}"):
                    if os.path.isfile(p):
                        os.remove(p)
                        nu+=1
        writelog(f"Removed {nu} files")
        # ---- OPT6: generate gifs
        if await_confirmation("generate gifs?") if "gif" not in config else config["gif"]:
            writelog("Generating GIFs...")
            power_gif(fdicts, gifdir, fps=5)
            writelog("GIFs Generated")
        # for method in ["npy","npy2","npy3","npy4"]:
        #     shutil.rmtree(str(Dirs.TEMP)+"/bindata", ignore_errors=True)
        #     dump_powerdicts(fdicts, method=method)
        #---- OPT7: Dump arrays to storage
        if await_confirmation("dump arrays to storage? (Required for histograms, but may take up large amount of storage)") if "dump_arrays" not in config else config["dump_arrays"]:
            fd = [{k:v for k,v in fdict.items() if k in ["visit","roi","fullim","datetime"]} for fdict in fdicts]
            writelog("Dumping Arrays...")
            dump_powerdicts(fd, method="npy")
            writelog("Arrays Dumped")
        #----! OPT8: Generate histograms (not implemented yet, but the plotting.dpr_histogram_plot() has been made)
        if await_confirmation("generate histograms?") if "histogram" not in config else config["histogram"]:
            opt=["Select which types to plot:", "average per visit", "overlaid per visit", ]
            ret = cutie.select_multiple(opt, [0], ticked_indices=[1])
            if len(ret) > 0:
                # list of visits in fdicts
                fdic = list(set([f["visit"] for f in fdicts]))
                writelog(f"Generating histograms for visits: {fdic}")
                pb12 = tqdm(total=len(ret)*len(fdic), desc="Generating Histograms")
                for r in ret:
                    if r<3:
                        methd = "avg" if r==1 else "indiv"
                        for visit in fdic:
                            found = glob(fpath(f"temp/bindata/{visit}_roi*"))
                            pb12.set_postfix_str(f"{visit} {methd}: {len(found)} arrays.")
                            dh = dpr_histogram_plot(found, method=methd)
                            fig, ax, d = dh
                            fig.suptitle("Histogram of Pixel Intensities within the DPR for visit "+visit)
                            fig.savefig(fpath(f"figures/imgs/{visit}_roi_hist_{methd}.png"))
                            pb12.update()
                            
                pb12.close()
                writelog("Histograms Generated")
                        

                      
        # ----! OPT9: Generate figures (not implemented yet, but the plotting functions have been made)
        if await_confirmation("Generate Figures?") if "figure" not in config else config["figure"]:
            
            def_figconfig= dict(
            stacked =   True,
            overlaid = True,
            megafigure = True,
            hst_cols = ["Total_Power","Avg_Flux","Area"],
            hisaki_cols = ["Torus_Power_Dawn","Torus_Power_Dusk","Pdyn","Aurora_Power","Aurora_Flux"],
            include = "last15")
            if fig_config.get("auto",False):
                fig_config = def_figconfig
            else:
                if "figtypes" not in fig_config:
                    opt = ["Select which plot types to generate:", "Overlaid", "Stacked", "Megafigure"]
                    inds =cutie.select_multiple(opt, [0], ticked_indices=[opt.index(i) for i in opt if i in def_figconfig.get("figtypes")])
                    fig_config["figtypes"] = [opt[i] for i in inds]
                fig_config.update({"stacked": "Stacked" in fig_config["figtypes"], "overlaid": "Overlaid" in fig_config["figtypes"], "megafigure": "Megafigure" in fig_config["figtypes"]})
                fig_config.remove("figtypes")
                if "default_cols" not in fig_config or fig_config.get("default_cols") is False:
                    if "hst_cols" not in fig_config: 
                        hst_cols = list(HST.colnames.values())
                        opt2 = ["Select which HST columns to plot:", *hst_cols, ]
                        fig_config["hst_cols"] = [opt2[i] for i in cutie.select_multiple(opt2, [0], ticked_indices=[i for i in opt2 if i in def_figconfig.get("hst_cols")])]
                    if "hisaki_cols" not in fig_config:
                        hisaki_cols = list(HISAKI.colnames.values())
                        opt3 = ["Select which Hisaki columns to plot:", *hisaki_cols, ]
                        fig_config["hisaki_cols"] = [opt3[i] for i in cutie.select_multiple(opt3, [0], ticked_indices=[i for i in opt3 if i in def_figconfig.get("hisaki_cols")])]
                else:
                    fig_config["hst_cols"] = def_figconfig.get("hst_cols")
                    fig_config["hisaki_cols"] = def_figconfig.get("hisaki_cols")
                
                if "include" not in fig_config:
                    opt4 = ["Select which datasets to plot:", "All", "Most Recent", "Last 5", "Last 10"]
                    fig_config["include"] = opt4[cutie.select(opt4, opt4.index(def_figconfig.get("include")))]
                
            writelog("Generating Figures...")
            
            if fig_config.get("stacked",False):
                writelog("Generating Stacked Figure")
                paths.extend(figure_gen(fig_config["hst_cols"],fig_config["hisaki_cols"],fig_config["include"],True,False,False))
                writelog("Stacked Figure Generated")
            if fig_config.get("overlaid",False):
                writelog("Generating Overlaid Figure")
                paths.extend(figure_gen(fig_config["hst_cols"],fig_config["hisaki_cols"],fig_config["include"],False,True,False))
                writelog("Overlaid Figure Generated")
            if fig_config.get("megafigure",False):
                writelog("Generating Megafigure")
                for h in fig_config["hst_cols"]:
                    writelog(f"Generating Megafigure for {h}")
                    paths.extend(figure_gen([h],fig_config["hisaki_cols"],fig_config["include"],False,False,True))
                writelog("Megafigure Generated")
            writelog("Figures Generated")
        if remove in ["all", "bindata"]:
            shutil.rmtree(str(Dirs.TEMP)+"/bindata", ignore_errors=True)
        if remove in ["all", "temp"]:
            shutil.rmtree(str(Dirs.TEMP), ignore_errors=True)
        if await_confirmation("Update Active Figures?") if "update" not in config else config["update"]:
            ensure_dir(fpath("figures/finished"))
            curr_files = os.listdir(fpath("figures/finished"))
            for p in paths:
                if p not in curr_files:
                    if "overlaid" in p:
                        shutil.copy(p, fpath("figures/finished/overlaid_plot.png"))
                    elif "stacked" in p:
                        shutil.copy(p, fpath("figures/finished/stacked_plot.png"))
                    elif "megafigure" in p:
                        col = p.split("[")[1].split("]")[0]
                        shutil.copy(p, fpath(f"figures/finished/megafigure_plot_{col}.png"))
                    # hist, gif
                
            
        
        
        writelog("Finished. Exiting...")
                
            
            
            
          
    run_path_powercalc(
        ignores={"groups": [], "visits": []},
        byvisit=True,
        segments=1,
        coadd_dir=fpath("temp/coadds"),
        avg_dir=fpath("temp/rollavgs"),
        gifdir=fpath("temp/gifs"),
        outfile="auto",
        extname="BOUNDARY",
        remove="none",
        window=5,
        config = {"generate_coadd":False, "coadd_pathfinder":False, "coadd_power":False, "generate_avgs":False, "avg_power":False, "avg_pathfinder":False, "gif":False, "histogram":False, "figure":True, "dump_arrays":False, "update":True},
        fig_config = {"auto":True}
    )


# loop to only generate the gaussians ----------------------------------------
#     for i in tqdm(gps):
#         basefitpath = fpath(f'datasets/HST/group_{i:0>2}')
#         fitsg = fits_from_glob(basefitpath)
#         copath = fpath(f'datasets/HST/custom/g{i:0>2},v{group_to_visit(i):0>2}_[3,1]gaussian-coadded.fits')
#         fit = generate_coadded_fits(fitsg, saveto=copath, gaussian=(3,1), overwrite=True,indiv=False, coadded=True)
#         fit.close()


    # writeto = f"{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}_powers_coadds.txt"
    # p = fpath("datasets/HST/custom/coadds/")
    
    # copath = fits_from_glob(p)
    # for cp in copath:
    #     if "BOUNDARY" in cp:
    #             pt = powercalc(cp, writeto=writeto)
        
        #pt = pathfinder(copath)
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
