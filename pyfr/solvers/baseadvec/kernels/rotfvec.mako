# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='rotfvec' ndim='2'
              vecs0='in fpdtype_t[${str(ndims)}]'
              vecs='out fpdtype_t[${str(ndims)}]'
              omg='scalar fpdtype_t'>

    // Rotation vector
    fpdtype_t rot[2][2] = {{cos(omg), -sin(omg)}, {sin(omg), cos(omg)}};

    // Multiply Rot Matrix
% for i in range(2):
    vecs[${i}] = 0.0;
% for j in range(2):
    vecs[${i}] += rot[${i}][${j}]*vecs0[${j}];
% endfor
% endfor
</%pyfr:kernel>