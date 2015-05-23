# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%
    import math
    pi = math.pi

    # Find nrow for amu_e
    if 'flux' in antialias:
        nrow = nqpts
    else:
        nrow = nupts
%>


<%pyfr:kernel name='avis' ndim='1'
              s='in view fpdtype_t[${str(nupts)}]'
              amu_e='out view fpdtype_t[${str(nrow)}]'
              amu_f='out view fpdtype_t[${str(nfpts)}]'>

    // Smoothness indicator
    fpdtype_t totEn = 0.0, pnEn = 1e-15, s2;
    fpdtype_t se0= ${math.log10(c['s0'])};

% for i, deg in enumerate(ubdegs):
    s2 = s[${i}]*s[${i}];
    totEn += s2;

% if deg >= order:
    pnEn += s2;
% endif
% endfor

    fpdtype_t se = log10(pnEn/totEn);

    // Compute cell-wise artificial viscosity
    fpdtype_t mu = (se < se0 - ${c['kappa']})
                 ? 0.0
                 : ${0.5*c['max-amu']}*(1.0 + sin(${0.5*pi/c['kappa']}*(se - se0)));
    mu = (se < se0 + ${c['kappa']}) ? mu : ${c['max-amu']};

    // Copy to all upts (or qpts) and fpts
% for i in range(nrow):
    amu_e[${i}] = mu;
% endfor

% for i in range(nfpts):
    amu_f[${i}] = mu;
% endfor
</%pyfr:kernel>
