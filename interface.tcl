#############################################################################
# Generated by PAGE version 5.4
#  in conjunction with Tcl version 8.6
#  Oct 10, 2020 02:53:08 PM -03  platform: Windows NT
set vTcl(timestamp) ""
if {![info exists vTcl(borrow)]} {
    tk_messageBox -title Error -message  "You must open project files from within PAGE."
    exit}


if {!$vTcl(borrow) && !$vTcl(template)} {

set vTcl(actual_gui_bg) #d9d9d9
set vTcl(actual_gui_fg) #000000
set vTcl(actual_gui_analog) #ececec
set vTcl(actual_gui_menu_analog) #ececec
set vTcl(actual_gui_menu_bg) #d9d9d9
set vTcl(actual_gui_menu_fg) #000000
set vTcl(complement_color) #d9d9d9
set vTcl(analog_color_p) #d9d9d9
set vTcl(analog_color_m) #ececec
set vTcl(active_fg) #000000
set vTcl(actual_gui_menu_active_bg)  #ececec
set vTcl(pr,menufgcolor) #000000
set vTcl(pr,menubgcolor) #d9d9d9
set vTcl(pr,menuanalogcolor) #ececec
set vTcl(pr,treehighlight) firebrick
set vTcl(pr,autoalias) 1
set vTcl(pr,relative_placement) 1
set vTcl(mode) Relative
}




proc vTclWindow.top44 {base} {
    global vTcl
    if {$base == ""} {
        set base .top44
    }
    if {[winfo exists $base]} {
        wm deiconify $base; return
    }
    set top $base
    ###################
    # CREATING WIDGETS
    ###################
    vTcl::widgets::core::toplevel::createCmd $top -class Toplevel \
        -background $vTcl(actual_gui_bg) 
    wm focusmodel $top passive
    wm geometry $top 1920x1017+651+187
    update
    # set in toplevel.wgt.
    global vTcl
    global img_list
    set vTcl(save,dflt,origin) 0
    wm maxsize $top 1924 1061
    wm minsize $top 120 1
    wm overrideredirect $top 0
    wm resizable $top 1 1
    wm title $top "New Toplevel"
    vTcl:DefineAlias "$top" "Toplevel1" vTcl:Toplevel:WidgetProc "" 1
    set vTcl(real_top) {}
    vTcl:withBusyCursor {
    canvas $top.can45 \
        -background $vTcl(actual_gui_bg) -borderwidth 2 -closeenough 1.0 \
        -height 856 -insertbackground black -relief ridge \
        -selectbackground blue -selectforeground white -width 1896 
    vTcl:DefineAlias "$top.can45" "Canvas1" vTcl:WidgetProc "Toplevel1" 1
    ttk::combobox $top.tCo46 \
        -font TkTextFont -textvariable combobox -foreground {} -background {} \
        -takefocus {} 
    vTcl:DefineAlias "$top.tCo46" "TCombobox1" vTcl:WidgetProc "Toplevel1" 1
    ttk::style configure TCheckbutton -background $vTcl(actual_gui_bg)
    ttk::style configure TCheckbutton -foreground $vTcl(actual_gui_fg)
    ttk::style configure TCheckbutton -font "$vTcl(actual_gui_font_dft_desc)"
    ttk::checkbutton $top.tCh47 \
        -variable tch47 -takefocus {} -text Tcheck 
    vTcl:DefineAlias "$top.tCh47" "TCheckbutton1" vTcl:WidgetProc "Toplevel1" 1
    scale $top.sca49 \
        -activebackground $vTcl(analog_color_m) \
        -background $vTcl(actual_gui_bg) -bigincrement 0.0 \
        -font TkDefaultFont -foreground $vTcl(actual_gui_fg) -from 0.0 \
        -highlightbackground $vTcl(actual_gui_bg) -highlightcolor black \
        -length 100 -orient horizontal -resolution 1.0 -tickinterval 0.0 \
        -to 100.0 -troughcolor #d9d9d9 
    vTcl:DefineAlias "$top.sca49" "Scale1" vTcl:WidgetProc "Toplevel1" 1
    ###################
    # SETTING GEOMETRY
    ###################
    place $top.can45 \
        -in $top -x 0 -relx 0.005 -y 0 -rely 0.069 -width 0 -relwidth 0.988 \
        -height 0 -relheight 0.842 -anchor nw -bordermode ignore 
    place $top.tCo46 \
        -in $top -x 0 -relx 0.005 -y 0 -rely 0.01 -width 0 -relwidth 0.988 \
        -height 0 -relheight 0.051 -anchor nw -bordermode ignore 
    place $top.tCh47 \
        -in $top -x 0 -relx 0.005 -y 0 -rely 0.934 -width 0 -relwidth 0.369 \
        -height 0 -relheight 0.047 -anchor nw -bordermode ignore 
    place $top.sca49 \
        -in $top -x 320 -y 940 -width 1587 -relwidth 0 -height 0 \
        -relheight 0.067 -anchor nw -bordermode ignore 
    } ;# end vTcl:withBusyCursor 

    vTcl:FireEvent $base <<Ready>>
}

set btop ""
if {$vTcl(borrow)} {
    set btop .bor[expr int([expr rand() * 100])]
    while {[lsearch $btop $vTcl(tops)] != -1} {
        set btop .bor[expr int([expr rand() * 100])]
    }
}
set vTcl(btop) $btop
Window show .
Window show .top44 $btop
if {$vTcl(borrow)} {
    $btop configure -background plum
}

