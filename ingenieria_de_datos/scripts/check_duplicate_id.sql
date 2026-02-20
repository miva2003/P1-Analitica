select estu_consecutivo, count(*)
from icfes_data
where cole_depto_ubicacion = 'CESAR'
group by estu_consecutivo
having count(*) > 1;

