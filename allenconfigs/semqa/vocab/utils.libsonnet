{
  boolparser(x)::
    if x == "true" then true
    else false
}

{
  parse_number(x)::
    local a = std.split(x, ".");
    if std.length(a) == 1 then
      std.parseInt(a[0])
    else
      local denominator = std.pow(10, std.length(a[1]));
      local numerator = std.parseInt(a[0] + a[1]);
      local parsednumber = numerator / denominator;
      parsednumber
}