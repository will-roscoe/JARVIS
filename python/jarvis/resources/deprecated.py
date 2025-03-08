#!/usr/bin/env python3
# ruff: noqa
# type: ignore


# DEPRECATED CODE, for reference or future use only
class PathFinder:
    _radioprops = {
        "facecolor": ["#000", _clickedkws["color"], _clickedkws["color"], _idpxkws["color"], _idpxkws["color"]],
        "edgecolor": ["#000", _clickedkws["color"], _clickedkws["color"], _idpxkws["color"], _idpxkws["color"]],
        "marker": ["o", "o", "X", "o", "X"],
    }
    _labelprops = {"color": ["#000", _clickedkws["color"], _clickedkws["color"], _idpxkws["color"], _idpxkws["color"]]}

    def __init__(
        self,
        fits_dir: fits.HDUList,
        saveloc=None,
        show_tooltips=True,
        morphex=(cv2.MORPH_CLOSE, cv2.MORPH_OPEN),
        fcmode=cv2.RETR_EXTERNAL,
        fcmethod=cv2.CHAIN_APPROX_SIMPLE,
        cvh=False,
        ksize=5,
        **persists,
    ):
        """### *JAR:VIS* Pathfinder Class
        > ***Requires PyQt6 (or PyQt5)***

        A GUI tool to select contours from a fits file. The tool allows the user to select luminosity samples from the image, and then to select a contour that encloses all of the selected samples. The user can then save the contour to the fits file, or to a new file if a save location is provided. The user can also change the morphological operations applied to the mask before finding the contours, the method used to find the contours, the method used to approximate the contours, and whether to use the convex hull of the contours. The user can also change the kernel size for the morphological operations.

        #### Parameters:
        - `fits_dir`: The fits file to open.
        - `saveloc`: The location to save the selected contour to. If `None`, the contour will be saved to the original fits file if the user chooses to save it.
        - `show_tooltips`: Whether to show tooltips when hovering over buttons.
        - Initial Config Options:
            - `morphex`: The morphological operations to apply to the mask before finding the contours. Default is `(cv2.MORPH_CLOSE,cv2.MORPH_OPEN)`.
            - `fcmode`: The method to find the contours. Default is `cv2.RETR_EXTERNAL`.
            - `fcmethod`: The method to approximate the contours. Default is `cv2.CHAIN_APPROX_SIMPLE`.
            - `cvh`: Whether to use convex hulls of the contours. Default is `False`.
            - `ksize`: The kernel size for the morphological operations. Default is `5`.

            These may be changed during the session using the GUI buttons.
        """
        raise NotImplementedError("This class is not yet implemented")
        mpl.use("QtAgg")
        from matplotlib.font_manager import fontManager

        fontManager.addfont()
        plt.rcParams["font.family"] = "FiraCode Nerd Font"
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["toolbar"] = "None"
        self.replace_fitsobj(fits_dir)
        self.CFG = dict(
            show_tooltips=show_tooltips, morphex=morphex, fcmode=fcmode, fcmethod=fcmethod, cvh=cvh, ksize=ksize
        )
        self.samples = dict(luminosity=persists.get("clicked_coords", []), identifiers=persists.get("identifiers", []))
        self.current_contour = persists.get("current_contour", None)
        self.fig = plt.figure()
        self.artists = dict()
        self.eventconns = dict()
        self.widgets = dict()

    def update_qtapp(self, fits_dir):
        fw = [[self.fig.canvas.width(), int(720 * 1.5)], [self.fig.canvas.height(), 720]]
        if any(i[0] < i[1] for i in fw):
            self.fig.canvas.manager.window.setGeometry(0, 0, fw[0][1], fw[1][1])
        self.fig.set_size_inches(fw[0][1] / self.fig.dpi, fw[1][1] / self.fig.dpi)
        self._qapp = self.fig.canvas.manager.window
        self._qapp.setWindowTitle(f"JAR:VIS Pathfinder ({fits_dir})")

        self._qapp.setWindowIcon(QtGui.QIcon(self._iconpath))

    def getareapct(self, contourarea):
        num = contourarea / self.IMAREA * 100
        if num < 0.01:
            return f"{num:.2e}%"
        return f"{num:.2f}%"

        # generate a stripped down, grey scale image of the fits file, and make normalised imagempl.rcParams['toolbar'] = 'None'

    def replace_fitsobj(self, fits_dir):
        self.read_dir = fits_dir
        self.fits_obj = fits.open(fits_dir, mode="update")
        proc = prep_polarfits(assign_params(self.fits_obj, fixed="LON", full=True))
        img = imagexy(proc, cmap=cmr.neutral, ax_background="white", img_background="black")
        self.normed = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        self.IMAREA = (np.pi * (self.normed.shape[0] / 2) ** 2) / 2
        self.update_qtapp(fits_dir)

    def init_axes(self):
        gs = self.fig.add_gridspec(1, 2, wspace=0.05, hspace=0, width_ratios=[8, 3], left=0, right=1, top=1, bottom=0)
        # mgs: 17 rows, 12 columns, to layout the options
        mgs = gs[1].subgridspec(
            11, 12, wspace=0, hspace=0, height_ratios=[0.5, 0.5, 0.5, 0.5, 1, 1, 0.3, 0.1, 1.5, 3, 3]
        )
        # layout the axes in the gridspecs
        self._gridspecs = {"plot": gs[0], "options": mgs}
        self._description_ax = self.fig.add_subplot(mgs[0:2, :8], label="DescriptionOutput")
        self._closebutton_ax = self.fig.add_subplot(mgs[0, 8:], label="CLOSE")
        self._savebutton_ax = self.fig.add_subplot(mgs[1, 8:], label="SAVE")
        self._resetbutton_ax = self.fig.add_subplot(mgs[2, 8:], label="RESET")
        self._frcreenbutton_ax = self.fig.add_subplot(mgs[3, 8:], label="FSCRN")
        self._clickselect_ax = self.fig.add_subplot(mgs[2:4, :8], label="ACTION", facecolor=_bgs["sidebar"])
        self._retrselect_ax = self.fig.add_subplot(mgs[4, :8], label="RETR", facecolor=_bgs["sidebar"])
        self._morphselect_ax = self.fig.add_subplot(mgs[4:6, 8:], label="MORPH", facecolor=_bgs["sidebar"])
        self._chainselect_ax = self.fig.add_subplot(mgs[5, :8], label="CHAIN", facecolor=_bgs["sidebar"])
        self._ksizeslider_ax = self.fig.add_subplot(mgs[7, :8], label="KSIZE", facecolor=_bgs["sidebar"])
        self._cvhselect_ax = self.fig.add_subplot(mgs[6:8, 8:], label="CHAIN", facecolor=_bgs["sidebar"])
        self._ksizevalue_ax = self.fig.add_subplot(mgs[6, :8], label="KSIZE", facecolor=_bgs["sidebar"])
        self._fitfile_ax = self.fig.add_subplot(mgs[8, :], label="fitinfo", facecolor=_bgs["sidebar"])
        self._contourlist_ax = self.fig.add_subplot(mgs[-1, :], label="legend", facecolor=_bgs["legend"])
        self._ax = self.fig.add_subplot(gs[0], label="main", zorder=11)
        if self.CFG["show_tooltips"]:
            self._tooltip_ax = self.fig.add_subplot(mgs[9, :], label="tooltip", facecolor="#000")
            self.artists["tootlip"] = self._tooltip_ax.text(
                0.012,
                0.98,
                "",
                fontsize=8,
                color="black",
                ha="left",
                va="top",
                wrap=True,
                bbox=dict(facecolor="white", alpha=0.5, lw=0),
                zorder=10,
            )
        self._axeslist = [
            self._ax,
            self._contourlist_ax,
            self._closebutton_ax,
            self._description_ax,
            self._retrselect_ax,
            self._morphselect_ax,
            self._chainselect_ax,
            self._clickselect_ax,
            self._savebutton_ax,
            self._cvhselect_ax,
            self._ksizeslider_ax,
            self._fitfile_ax,
            self._resetbutton_ax,
            self._ksizevalue_ax,
            self._frcreenbutton_ax,
        ] + ([self._tooltip_ax] if self.CFG["show_tooltips"] else [])
        self._axgroups = {
            "buttons": [self._closebutton_ax, self._savebutton_ax, self._resetbutton_ax, self._frcreenbutton_ax],
            "textboxes": [self._description_ax, self._fitfile_ax],
            "options": [
                self._retrselect_ax,
                self._morphselect_ax,
                self._chainselect_ax,
                self._cvhselect_ax,
                self._ksizeslider_ax,
                self._ksizevalue_ax,
                self._clickselect_ax,
            ],
            "legends": [self._contourlist_ax],
        }

    def update_fitdesc(self):
        self._fitfile_ax.clear()
        visit, tel, inst, filt, pln, hem, cml, doy = (
            *fitsheader(self.fits_obj, "VISIT", "TELESCOP", "INSTRUME", "FILTER", "PLANET", "HEMISPH", "CML", "DOY"),
        )
        dt = get_datetime(self.fits_obj)
        self._fitfile_ax.text(0.04, 0.96, "FITS File Information", fontsize=12, color="black", ha="left", va="top")
        pth = str(self.read_dir)
        pthparts = []
        if len(pth) > 30:
            parts = split_path(rpath(pth))
            while len(parts) > 0:
                stick = parts.pop(-1)
                if len(parts) > 0:
                    if len(stick) + len(parts[-1]) < 30:
                        parts[-1] += stick
                    else:
                        pthparts.append(stick)
                else:
                    pthparts.append(stick)
        else:
            pthparts.append(pth)
        pthparts.reverse()
        ctxt = [
            ["File", f"{pthparts[0]}"],
            *[[" ", f"{p}"] for p in pthparts[1:]],
            ["Inst.", f"{inst} ({tel}, {filt} filter)"],
            ["Obs.", f"visit {visit} of {pln} ({hem})"],
            ["Date", f'{dt.strftime("%d/%m/%Y")}, day {doy}'],
            ["Time", f'{dt.strftime("%H:%M:%S")}(ICRS)'],
            ["CML", f"{cml:.4f}°"],
        ]
        tl = self._fitfile_ax.table(
            cellText=ctxt,
            cellLoc="right",
            cellColours=[[_bgs["sidebar"], _bgs["sidebar"]] for i in range(len(ctxt))],
            colWidths=[0.14, 1],
            colLabels=None,
            rowLabels=None,
            colColours=None,
            rowColours=None,
            colLoc="center",
            rowLoc="center",
            loc="bottom left",
            bbox=[0.02, 0.02, 0.96, 0.8],
            zorder=0,
        )
        tl.visible_edges = ""
        tl.auto_set_font_size(False)
        tl.set_fontsize(9)
        for key, cell in tl.get_celld().items():
            cell.set_linewidth(0)
            cell.PAD = 0
            if key[0] < len(pthparts) and key[1] == 1:
                cell.set_fontsize(10 - len(pthparts) + 1)
        self.fig.canvas.blit(self._fitfile_ax.bbox)

    def update_descriptor(self, *args, **kwargs):
        self._description_ax.clear()
        if len(args) >= 2:
            x, y = args[:2]
        else:
            x, y = 0.5, 0.5
        if len(args) == 3:
            st = args[2]
        elif len(args) == 1:
            st = args
        else:
            st = ""
        kwargs.setdefault("fontsize", 10)
        kwargs.setdefault("color", "black")
        kwargs.setdefault("ha", "center")
        kwargs.setdefault("va", "center")
        self._description_ax.text(x, y, st, **kwargs)
        self.fig.canvas.blit(self._description_ax.bbox)

    def initialise_app(self):
        self._ksizeslider_ax.set_facecolor("#fff")
        for a in self._axeslist:
            a.xaxis.set_visible(False)
            a.yaxis.set_visible(False)
        self.fig.canvas.set_cursor(3)
        self.fig.set_facecolor("black")
        self.update_fitdesc(self.fits_obj)

    def update_mainax(self):
        for a in [self._ax, self._contourlist_ax, self._description_ax]:
            a.clear()
            a.set_facecolor(get_bg(a))
            self._ax.imshow(self.normed, cmap=cmr.neutral, zorder=0)
        if len(self.samples["identifiers"]) > 0:
            pxsc = (
                [idpixels[0] for idpixels in self.samples["identifiers"]],
                [idpixels[1] for idpixels in self.samples["identifiers"]],
            )
            self._ax.scatter(*pxsc, **_idpxkws)
        if len(self.samples["luminosity"]) > 0:
            scpc = [c[1][0] for c in self.samples["luminosity"]], [c[1][1] for c in self.samples["luminosity"]]
            self._ax.scatter(*scpc, **_clickedkws)
        if len(self.samples["luminosity"]) > 1:
            self.lrange = [min(c[0] for c in self.samples["luminosity"]), max(c[0] for c in self.samples["luminosity"])]
            self.mask = cv2.inRange(self.normed, self.lrange[0] * 255, self.lrange[1] * 255)
            kernel = np.ones((self.CFG["ksize"], self.CFG["ksize"]), np.uint8)
            for morph in self.CFG["morphex"]:
                self.mask = cv2.morphologyEx(self.mask, morph, kernel)
            contours, hierarchy = cv2.findContours(
                image=self.mask, mode=self.CFG["fcmode"], method=self.CFG["fcmethod"]
            )
            if self.CFG["cvh"]:
                contours = [cv2.convexHull(cnt) for cnt in contours]
            sortedcs = sorted(contours, key=lambda x: cv2.contourArea(x))
            sortedcs.reverse()

            # update the description to show the lrange on row 1

            if len(self.samples["identifiers"]) > 0:
                selected_contours = []
                other_contours = []
                for i, contour in enumerate(sortedcs):
                    if all(
                        [cv2.pointPolygonTest(contour, id_pixel, False) > 0 for id_pixel in self.samples["identifiers"]]
                    ):
                        selected_contours.append([i, contour])
                    else:
                        other_contours.append([i, contour])
                for i, contour in selected_contours:
                    self._ax.scatter(*contour.T, **_selectclinekws)
                    cpath = mpl.path.Path(contour.reshape(-1, 2))
                    self._ax.plot(*cpath.vertices.T, color=_selectclinekws["color"], lw=_selectclinekws["s"])
                    self._ax.text(contour[0][0][0], contour[0][0][1], f"{i}", **_selectctextkws)
                for i, contour in other_contours:
                    self._ax.scatter(*contour.T, **_otherclinekws)
                    cpath = mpl.path.Path(contour.reshape(-1, 2))
                    self._ax.plot(*cpath.vertices.T, color=_otherclinekws["color"], lw=_otherclinekws["s"])
                    self._ax.text(contour[0][0][0], contour[0][0][1], f"{i}", **_otherctextkws)

                # update the description output to show the number of contours on row 2

                if len(selected_contours) > 0:
                    # update the description output to show the area of the selected contour on row 3
                    self.current_contour = selected_contours[0][1]
                selectedhandles = [
                    mpl.lines.Line2D(
                        [0], [0], label=f"{i}, {self.getareapct(cv2.contourArea(sortedcs[i]))}", **_handles["selectedc"]
                    )
                    for i in [c[0] for c in selected_contours]
                ]
                otherhandles = [
                    mpl.lines.Line2D(
                        [0], [0], label=f"{i}, {self.getareapct(cv2.contourArea(sortedcs[i]))}", **_handles["otherc"]
                    )
                    for i in [c[0] for c in other_contours]
                ]
                self._contourlist_ax.legend(handles=selectedhandles + otherhandles, **_legendkws)
            else:
                for i, contour in enumerate(sortedcs):
                    self._ax.scatter(*contour.T, **_defclinekws)
                    cpath = mpl.path.Path(contour.reshape(-1, 2))
                    self._ax.plot(*cpath.vertices.T, color=_defclinekws["color"], lw=_defclinekws["s"])
                    self._ax.text(contour[0][0][0], contour[0][0][1], f"{i}", **_defctextkws)
                self._contourlist_ax.text(
                    0.5, 0.5, f"Contours: {len(sortedcs)}", fontsize=10, color="black", ha="center", va="center"
                )
                handles = [
                    mpl.lines.Line2D(
                        [0], [0], label=f"{i}, {self.getareapct(cv2.contourArea(sortedcs[i]))}", **_handles["defc"]
                    )
                    for i in range(len(sortedcs))
                ]
                self._contourlist_ax.legend(handles=handles, **_legendkws)
                self._contourlist_ax.set_facecolor(get_bg(self._contourlist_ax))
        else:
            # update the description output to show the prompt to select luminosity samples
            self._contourlist_ax.set_facecolor(get_bg(self._ax))
        self.fig.canvas.blit(self._ax.bbox)
        self.fig.canvas.blit(self._contourlist_ax.bbox)

    def _widget_reset_button(self, event):
        self.samples = dict(luminosity=[], identifiers=[])
        self.update_mainax()

    def _widget_close_button(self, event):
        plt.close()

    def _widget_fullscreen_button(self, event):
        self.fig.canvas.manager.full_screen_toggle()

    def _widget_save_button(self, event):
        if self.current_contour is not None:
            pth = self.current_contour.reshape(-1, 2)
            pth = fullxy_to_polar_arr(pth, self.normed, 40)
            if self.saveloc is not None:  # if a save location is provided, save the contour to a new fits file
                n_fits_obj = save_contour(fits_obj=self.fits_obj, cont=pth)
                n_fits_obj.writeto(self.saveloc, overwrite=True)
                tqdm.write(f"Saved to {self.saveloc}, closing")
                self.update_descriptor("Saved to " + str(self.saveloc) + "\n, closing")
                time.sleep(1)
                plt.close()
                self.fits_obj.close()

            else:
                self.fits_obj.append(contourhdu(pth))
                self.fits_obj.flush()
                tqdm.write(f"Save Successful, contour added to fits file at index {len(self.fits_obj)-1}")
                self.update_descriptor(
                    "Save Successful, contour added\n to fits file at index" + str(len(self.fits_obj) - 1)
                )
        else:
            tqdm.write("Save Failed: No contour selected to save")
            self.update_descriptor("Save Failed: No contour selected to save")

    def _widget_retr_select(self, label):
        self.CFG["fcmode"] = _cvtrans["RETR"][label][0]

    def _widget_morph_select(self, lb):
        if _cvtrans["MORPH"][lb][0] in self.CFG["morphex"]:
            self.CFG["morphex"].remove(_cvtrans["MORPH"][lb][0])
        else:
            self.CFG["morphex"].append(_cvtrans["MORPH"][lb][0])

    def _widget_chain_select(self, label):
        self.CFG["fcmethod"] = _cvtrans["CHAIN"][label][0]

    def _widget_cvh_select(self, label):
        self.CFG["cvh"] = not self.CFG["cvh"]

    def _widget_ksize_slider(self, val):
        self._ksizeslider_ax.set_facecolor("#fff")
        self.CFG["ksize"] = int(val)
        self._ksizevalue_ax.clear()
        self._ksizevalue_ax.text(
            0.1, 0.45, f'ksize: {self.CFG["ksize"]}', fontsize=10, color="black", ha="left", va="center"
        )
        self.fig.canvas.blit(self._ksizevalue_ax.bbox)
        # self.update_mainax()

    def _widget_click_select(self, label):
        if "Luminosity" in label:
            ret = 1
        elif "IDPX" in label:
            ret = 2
        if "Remove" in label:
            ret = -ret
        self.G_clicktype = ret

    def _on_click(self, event):
        if event.inaxes == self._ax:
            if event.xdata is not None and event.ydata is not None:
                click_coords = (event.xdata, event.ydata)
                if self.G_clicktype == 1:
                    self.samples["luminosity"].append(
                        [self.normed[int(event.ydata), int(event.xdata)] / 255, click_coords]
                    )
                elif self.G_clicktype == -1:
                    if len(self.samples["luminosity"]) > 0:
                        self.samples["luminosity"].pop(
                            np.argmin(
                                [
                                    np.linalg.norm(np.array(c[1]) - np.array(click_coords))
                                    for c in self.samples["luminosity"]
                                ]
                            )
                        )
                elif self.G_clicktype == 2:
                    self.samples["identifiers"].append(click_coords)
                elif self.G_clicktype == -2:
                    if len(self.samples["identifiers"]) > 0:
                        self.samples["identifiers"].pop(
                            np.argmin(
                                [
                                    np.linalg.norm(np.array(c) - np.array(click_coords))
                                    for c in self.samples["identifiers"]
                                ]
                            )
                        )
        self.update_mainax()

    def _on_move(self, event):
        inax = event.inaxes
        if inax in self._axgroups["options"] + self._axgroups["buttons"]:
            axlabel = inax.get_label()
            txts = [t for t in inax.get_children() if isinstance(t, mpl.text.Text)]
            txts = [t for t in txts if t.get_text() != ""]
            txt = None
            for t in txts:
                if t.contains(event)[0]:
                    txt = t
                    break
            if txt is not None:
                item = txt.get_text()
                try:
                    texti = [f"{axlabel}.{item}", _cvtrans[axlabel][item][1]]
                except KeyError:
                    texti = [f"{axlabel}", _cvtrans[axlabel]["info"]]

            else:
                texti = [axlabel, _cvtrans[axlabel]["info"]]
            self.artists["tootlip"].set_text(f"{texti[0]}: {texti[1]}")
            self.fig.canvas.draw_idle()
            self.fig.canvas.blit(self._tooltip_ax.bbox)

    def _on_key(self, event):
        _keybindings = {
            "0": (self._widget_click_select, "None"),
            "1": (self._widget_click_select, "Add Luminosity"),
            "2": (self._widget_click_select, "Add IDPX"),
            "3": (self._widget_click_select, "Remove Luminosity"),
            "4": (self._widget_click_select, "Remove IDPX"),
            "s": (self._widget_save_button, "none"),
            "r": (self._widget_reset_button, "none"),
            "f": (self._widget_fullscreen_button, "none"),
            "q": (self._widget_close_button, "none"),
        }
        if event.key in _keybindings:
            action, val = _keybindings[event.key]
            action(val)

    def init_widgets(self):
        widgets = dict(
            RESET=mpl.widgets.Button(self._resetbutton_ax, "Reset      " + "\uf0e2"),
            CLOSE=mpl.widgets.Button(self._closebutton_ax, "Close      " + "\uf011"),
            SAVE=mpl.widgets.Button(self._savebutton_ax, "Save      " + "\uf0c7"),
            FSCREEN=mpl.widgets.Button(self._frcreenbutton_ax, "Fullscreen " + "\uf50c"),
            RETR=mpl.widgets.RadioButtons(self._retrselect_ax, ["EXTERNAL", "LIST", "CCOMP", "TREE"], active=0),
            MORPH=mpl.widgets.RadioButtons(
                self._morphselect_ax,
                *list(
                    zip(
                        [
                            [k, True if k in self.CFG["morphex"] else False]
                            for k in list(_cvtrans["MORPH"].keys() - ["info"])
                        ]
                    )
                ),
            ),
            CHAIN=mpl.widgets.RadioButtons(self._chainselect_ax, ["NONE", "SIMPLE", "TC89_L1", "TC89_KCOS"], active=1),
            CVH=mpl.widgets.RadioButtons(self._cvhselect_ax, ["Convex Hull"], active=[self.CFG["cvh"]]),
            CLICK=mpl.widgets.RadioButtons(
                self._clickselect_ax,
                ["Add Luminosity", "Remove Luminosity", "Add IDPX", "Remove IDPX"],
                active=0,
                radio_props=self._radioprops,
                label_props=self._labelprops,
            ),
            KSIZE=mpl.widgets.Slider(
                self._ksizeslider_ax, valmin=1, valmax=11, valinit=G_ksize, valstep=2, color="orange", initcolor="None"
            ),
        )
        widgets["KSIZE"].add_patch(
            mpl.patches.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="#fff",
                lw=1,
                zorder=-1,
                transform=self._ksizeslider_ax.transAxes,
                edgecolor="black",
            )
        )
        widgets["KSIZE"].add_patch(
            mpl.patches.Rectangle(
                (0, 0.995), 5 / 12, 1.005, facecolor="#fff", lw=0, zorder=10, transform=self._ksizeslider_ax.transAxes
            )
        )
        conns = dict(
            RESET=widgets["RESET"].on_clicked(self._widget_reset_button),
            CLOSE=widgets["CLOSE"].on_clicked(self._widget_close_button),
            SAVE=widgets["SAVE"].on_clicked(self._widget_save_button),
            FSCREEN=widgets["FSCREEN"].on_clicked(self._widget_fullscreen_button),
            RETR=widgets["RETR"].on_clicked(self._widget_retr_select),
            MORPH=widgets["MORPH"].on_clicked(self._widget_morph_select),
            CHAIN=widgets["CHAIN"].on_clicked(self._widget_chain_select),
            CVH=widgets["CVH"].on_clicked(self._widget_cvh_select),
            CLICK=widgets["CLICK"].on_clicked(self._widget_click_select),
            KSIZE=widgets["KSIZE"].on_changed(self._widget_ksize_slider),
        )
        self.widgets = widgets
        self.eventconns.update(widgets=conns)

    def init_events(self):
        move_event = self.fig.canvas.mpl_connect("motion_notify_event", self._on_move)
        click_event = self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        key_event = self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.eventconns.update(move=move_event, click=click_event, keypress=key_event)

    def run(self):
        self.init_axes()
        self.initialise_app()
        self.init_widgets()
        self.init_events()
        plt.show()
        self.fits_obj.close()
        return self.samples, self.current_contour


class Jfits:
    ind = FITSINDEX

    def __init__(self, fits_loc: str = None, fits_obj: fits.HDUList = None, **kwargs):
        if fits_loc is not None:
            self.loc = fits_loc
            self.hdul = fits.open(fits_loc)
        else:
            self.hdul = fits_obj
        self.header.update(kwargs)

    @property
    def data(self):
        return self.hdul[self.ind].data

    @property
    def header(self):
        return self.hdul[self.ind].header

    def update(self, data=None, **kwargs):
        if data is not None:
            self.hdul = adapted_fits(self.hdul, new_data=data)
        for k, v in kwargs.items():
            self.hdul = adapted_fits(self.hdul, **{k: v})

    def writeto(self, path: str):
        self.hdul.writeto(path)

    def close(self):
        self.hdul.close()

    def data_apply(self, func, *args, **kwargs):
        self.update(data=func(self.data, *args, **kwargs))

    def apply(self, func, *args, **kwargs):
        self = func(self, *args, **kwargs)

    def __del__(self):
        self.close()
