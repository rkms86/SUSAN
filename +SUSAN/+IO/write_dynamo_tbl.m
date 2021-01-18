function write_dynamo_tbl(tbl,tablename)
% WRITE_DYNAMO_TBL Writes a DYNAMO table.
%   WRITE_DYNAMO_TBL(TBL,TABLENAME) Saves TBL as a DYNAMO table with the
%   name TABLENAME.
%
%   See also dthelp.

dlmwrite(tablename,real(tbl),' ');

end
