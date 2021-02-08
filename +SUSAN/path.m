function susan_path = path()
% PATH returns the current SUSAN path.
%   SUSAN_PATH= PATH() returns the current SUSAN path

what_susan = what('SUSAN');
susan_path = what_susan(1).path;

end