# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='entropy' ndim='2'
              u='in fpdtype_t[${str(nvars)}]'
              s='out fpdtype_t'>
    // Compute the velocities
    fpdtype_t invrho = 1.0/u[0], E = u[${nvars - 1}];
    fpdtype_t p, rhov[${ndims}];
% for i in range(ndims):
    rhov[${i}] = u[${i + 1}];
% endfor

    // Compute the pressure
    p = ${c['gamma'] - 1}*(E - 0.5*invrho*${pyfr.dot('rhov[{i}]', i=ndims)});

    // Compute Entropy
    s = p*pow(invrho, ${c['gamma']});
</%pyfr:kernel>
