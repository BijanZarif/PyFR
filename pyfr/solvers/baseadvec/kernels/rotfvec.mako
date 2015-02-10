# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='rotfvec' ndim='2'
              vecs='inout fpdtype_t[${str(ndims)}]'
              omg='scalar fpdtype_t'>

    fpdtype_t tmp[${str(ndims)}];

    // Rotation vector
    fpdtype_t rot[2][2] = {{cos(omg), -sin(omg)}, {sin(omg), cos(omg)}};

    // Multiply Rot Matrix
% for i in range(2):
    tmp[${i}] = 0.0;
% for j in range(2):
    tmp[${i}] += rot[${i}][${j}]*vecs[${j}];
% endfor
% endfor

   // Copy matrix
% for i in range(2):
    vecs[${i}] = tmp[${i}];
% endfor
</%pyfr:kernel>