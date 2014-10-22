# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.flux'/>

//RoeM scheme (ref: JCP 185(2), 342-374)
<%pyfr:macro name='rsolve' params='ul, ur, n, nf'>
    // Compute the left and right fluxes + velocities and pressures
    fpdtype_t fl[${ndims}][${nvars}], fr[${ndims}][${nvars}];
    fpdtype_t vl[${ndims}], vr[${ndims}], va[${ndims}], dv[${ndims}];
    fpdtype_t du[${nvars}], bdq[${nvars}];
    fpdtype_t pl, pr;

    ${pyfr.expand('inviscid_flux', 'ul', 'fl', 'pl', 'vl')};
    ${pyfr.expand('inviscid_flux', 'ur', 'fr', 'pr', 'vr')};

    //specific enthalpy, contra velocity for left / right
    fpdtype_t hl = (ul[${ndims + 1}] + pl)/ul[0];
    fpdtype_t hr = (ur[${ndims + 1}] + pr)/ur[0];
    fpdtype_t contral = ${pyfr.dot('n[{i}]', 'vl[{i}]', i=ndims)};
    fpdtype_t contrar = ${pyfr.dot('n[{i}]', 'vr[{i}]', i=ndims)};

    //Difference between two state
    fpdtype_t drho = ur[0] - ul[0];
    fpdtype_t dp   = pr - pl;
    fpdtype_t dh   = hr - hl;
    fpdtype_t dcontra = contrar - contral;
% for i in range(ndims):
    dv[${i}] = vr[${i}] - vl[${i}];
% endfor

    // Compute Roe averaged density and enthalpy
    fpdtype_t rrr  = sqrt(ur[0]/ul[0]);
    fpdtype_t ratl = 1.0/(1.0 + rrr);
    fpdtype_t ratr = rrr*ratl;
    fpdtype_t ra   = rrr*ul[0];
    fpdtype_t ha   = hl*ratl + hr*ratr;

% for i in range(ndims):
    va[${i}] = (vl[${i}]*ratl + vr[${i}]*ratr);
% endfor

    fpdtype_t qq      = ${pyfr.dot('va[{i}]', 'va[{i}]', i=ndims)};
    fpdtype_t contraa = ${pyfr.dot('n[{i}]', 'va[{i}]', i=ndims)};
    fpdtype_t aa      = sqrt(${c['gamma'] - 1}*(ha - 0.5*qq));
    fpdtype_t ma      = contraa/aa;

    //Eigen structure
    fpdtype_t b1 = max(0.0, max(contraa + aa, contral + aa));
    fpdtype_t b2 = min(0.0, min(contraa - aa, contrar - aa));

    // 1-D shock discontinuity sensing term and Mach number based function f,g
    fpdtype_t SDST = pl/pr;
    SDST = min(SDST, 1/SDST);

    fpdtype_t h = 1.0 - SDST;
    fpdtype_t f = pow(abs(ma), h);
    fpdtype_t g = f;

    //du
%for i in range(nvars - 1):
    du[${i}] = ur[${i}] - ul[${i}];
%endfor
    du[${nvars}-1] = ur[0]*hr - ul[0]*hl;

    //BdQ
    bdq[0] = drho - f*dp/(aa*aa);
    bdq[${nvars}-1] = bdq[0]*ha + ra*dh;
% for i in range(ndims):
    bdq[${i}+1] = bdq[0]*va[${i}] + ra*(dv[${i}] - n[${i}]*dcontra);
% endfor

    //flux
% for i in range(nvars):
    nf[${i}] = (${' + '.join('n[{j}]*(b1*fl[{j}][{i}] - b2*fr[{j}][{i}])'
                .format(i=i, j=j) for j in range(ndims))}
                + b1*b2*(du[${i}] - g/(1.0 + abs(ma))*bdq[${i}]))/(b1 - b2);
% endfor
</%pyfr:macro>
