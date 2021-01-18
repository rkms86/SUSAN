function tbl = read_dynamo_tbl(tablename)
% READ_DYNAMO_TBL Reads a DYNAMO table.
%   TBL = READ_DYNAMO_TBL(TABLENAME) Reads TABLENAME as a DYNAMO table.
%
%   See also dthelp.

tbl = dlmread(tablename);

end
