# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='avis' ndim='1'
              s='inout fpdtype_t'
              amu_e='out fpdtype_t'
              amu_f='out fpdtype_t'>

    // Pointer for entropy, artificial viscosity on upts/fpts
    fpdtype_t* s_p = &s;
    fpdtype_t* amu_e_p = &amu_e;
    fpdtype_t* amu_f_p = &amu_f;
    fpdtype_t s_e[${nupts}];

    // Modal of entropy, energy of mode
    fpdtype_t totEn = 0.0, pnEn = 0.0;
% for i in range(nupts):
    s_e[${i}] = *(s_p + ${i*lds});
    totEn += s_e[${i}]*s_e[${i}];
% if ubdegs[i] >= order:
    pnEn += s_e[${i}]*s_e[${i}];
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
    *(amu_e_p + ${i*lds}) = mu;
% endfor

% for i in range(nfpts):
    *(amu_f_p + ${i*lds}) = mu;
% endfor

</%pyfr:kernel>
