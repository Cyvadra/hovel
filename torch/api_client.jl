
using HDF5
X = Float32.(collect(hcat(wholeX...)'))
Y = Float32.(collect(hcat(wholeY...)'))
# Apply custom metrics
lambda(y) = abs(y) <= 1.0 ? y : sign(y) * (1 + log(abs(y)))
Y = map(x->lambda(x), Y)
h5open("training_data.0805.h5", "w") do file
  write(file, "X", X)
  write(file, "Y", Y)
  end


using HTTP, JSON

function lplot(v,rng)
  a=lineplot(v;width=140,height=31,ylim=(rng[1],rng[end]))
  return a
  end
  
function remotePredict(multiple_x::Vector{Vector{T}})::Vector{Vector} where T <: Real
  res = HTTP.post("http://172.16.1.100:8084/predict", [], json(Dict("data"=>multiple_x))).body |> String |> JSON.Parser.parse
  return res["predictions"]

function randPlot()
  minI = ceil(Int,length(wholeY)*0.9)
  maxI = length(wholeY)
  i = rand(minI:maxI)
  p = remotePredict([wholeX[i]])[1]
  tmpList = vcat(p, wholeY[i])
  @info min(tmpList...), max(tmpList...)
  lineplot!(lplot(p, [reduce(min,tmpList)-0.5,0.5+reduce(max,tmpList)]), wholeY[i])
  end

function randPlotPredictOnly()
  minI = ceil(Int,length(wholeY)*0.9)
  maxI = length(wholeY)
  i = rand(minI:maxI)
  p = remotePredict([wholeX[i]])[1]
  tmpList = p
  lplot(p, [reduce(min,tmpList)-0.5,0.5+reduce(max,tmpList)])
  end

function randPlotTrainingSet()
  i = rand(1:round(Int,length(wholeY)*0.9))
  p = remotePredict([wholeX[i]])[1]
  tmpList = vcat(p, wholeY[i])
  @info min(tmpList...), max(tmpList...)
  lineplot!(lplot(p, [reduce(min,tmpList)-0.5,0.5+reduce(max,tmpList)]), wholeY[i])
  end

function randPlotTrainingSetPredictOnly()
  i = rand(1:round(Int,length(wholeY)*0.9))
  p = remotePredict([wholeX[i]])[1]
  tmpList = p
  lplot(p, [reduce(min,tmpList)-0.5,0.5+reduce(max,tmpList)])
  end
