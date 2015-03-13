# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='viewcopy' ndim='1'
              ul='in view fpdtype_t[${str(nvars)}]'
              ur='out view fpdtype_t[${str(nvars)}]'>

% for i in range(nvars):
    ur[${i}] = ul[${i}];
% endfor
</%pyfr:kernel>