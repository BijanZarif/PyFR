# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='rotsmat' ndim='2'
              smats0='in fpdtype_t[${str(ndims)}][${str(ndims)}]'
              smats='out fpdtype_t[${str(ndims)}][${str(ndims)}]'
              omg='scalar fpdtype_t'>

    // Rotation vector (transpose)
    fpdtype_t rot[2][2] = {{cos(omg), sin(omg)}, {-sin(omg), cos(omg)}};

    // Multiply Rot Matrix
% for i, j in pyfr.ndrange(ndims, ndims):
    smats[${i}][${j}] = 0.0;
% for k in range(2):
    smats[${i}][${j}] += smats0[${i}][${k}]*rot[${k}][${j}];
% endfor
% endfor
</%pyfr:kernel>