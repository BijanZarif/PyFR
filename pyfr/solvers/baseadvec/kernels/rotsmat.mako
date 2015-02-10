# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='rotsmat' ndim='2'
              smats='inout fpdtype_t[${str(ndims)}][${str(ndims)}]'
              omg='scalar fpdtype_t'>

    fpdtype_t tmp[${str(ndims)}][${str(ndims)}];

    // Rotation vector (transpose)
    fpdtype_t rot[2][2] = {{cos(omg), sin(omg)}, {-sin(omg), cos(omg)}};

    // Multiply Rot Matrix
% for i, j in pyfr.ndrange(ndims, ndims):
    tmp[${i}][${j}] = 0.0;
% for k in range(2):
    tmp[${i}][${j}] += smats[${i}][${k}]*rot[${k}][${j}];
% endfor
% endfor

   // Copy matrix
% for i, j in pyfr.ndrange(ndims, ndims):
    smats[${i}][${j}] = tmp[${i}][${j}];
% endfor
</%pyfr:kernel>