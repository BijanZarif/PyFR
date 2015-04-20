# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='avis' ndim='1'
              s='out view fpdtype_t[${str(nupts)}]'
              amu_e='out view fpdtype_t[${str(nupts)}]'
              amu_f='out view fpdtype_t[${str(nfpts)}]'>

    // Modal of entropy, energy of mode
    fpdtype_t totEn = 0.0, pnEn = 0.0;
% for i in range(nupts):
    totEn += s[${i}]*s[${i}];
% if ubdegs[i] >= order:
    pnEn += s[${i}]*s[${i}];
% endif
% endfor

    // Sensor for artificial viscosity
    fpdtype_t mu;
    fpdtype_t pi = 4.0*atan(1.0);
    fpdtype_t se = log10(pnEn/totEn + 1e-15);
    fpdtype_t se0= log10(${c['s0']});

    mu = (se < se0 - ${c['kappa']}) ? 0.0 : ${c['amu0']}*0.5*(1.0 + sin(pi*(se - se0)/(2*${c['kappa']})));
    mu = (se < se0 + ${c['kappa']}) ? mu : ${c['amu0']};

    // Copy to all spts/fpts
% for i in range(nupts):
    amu_e[${i}] = mu;
% endfor

% for i in range(nfpts):
    amu_f[${i}] = mu;
% endfor

</%pyfr:kernel>
