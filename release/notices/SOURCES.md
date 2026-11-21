# Windows distribution notices

The build copies Python package notices from the pinned shell environment, React production licenses from the lockfile installation and CPython's combined distribution license from the build interpreter. This directory also accompanies the full license and notice from the matching Microsoft.Web.WebView2 1.0.3856.49 SDK and both uv 0.12.13 licenses.

The Tcl notice is from the Tcl 8.6 release. The proxy_tools wheel omits its upstream license file. The build includes that file alongside the package notices. [Tcl license](https://github.com/tcltk/tcl/blob/core-8-6-14/license.terms) · [proxy_tools license](https://github.com/jtushman/proxy_tools/blob/master/LICENSE.txt)

WebView2 Evergreen runs as a separately installed Microsoft prerequisite. The installer uses Microsoft's runtime detection and signed bootstrapper. [WebView2 distribution](https://learn.microsoft.com/microsoft-edge/webview2/concepts/distribution)
