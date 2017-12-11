# arima实验对比
setwd("E:\\disk_use_201707")
# caorui@nbjl.nankai.edu.cn
# start 2017/12/04 23:14 v1.0.0
# end 
library("forecast")
arima_model <- function(data_in, pre_length)
{
  fit <- auto.arima(as.numeric(line))
  # plot(forecast(fit, h = pre_length))
  return(forecast(fit, h = pre_length))
}

cal_error <- function(pre, ori)
{
  result_error <- NULL
  for(index_col in c(1 : ncol(pre)))
  {
    TP = 0  # BOTH 0,
    FP = 0  # PRE 0, ACT 1
    FN = 0  # PRE 1, ACT 0
    TN = 0  # BOTH 1
    # disk >75, CPU < 20, MEM > 80
    for(index_row in c(1 : nrow(pre)))
    {
      if(ori[index_row, index_col] == 1)
      {
        if(pre[index_row, index_col] > 80)
        {
          TN = TN + 1
        }else
        {
          FP = FP + 1
        }
      }else
      {
        if(pre[index_row, index_col] > 80)
        {
          FN = FN + 1
        }else
        {
          TP = TP + 1
        }
      }
    }
    alarm_rate = TN / (TN + FP)
    error_rate = (FN + FP) / (TP + FN + TN + FP)
    print(c(alarm_rate, error_rate))
    error <- c(alarm_rate, error_rate)
    result_error <- rbind(result_error, error)
  }
  return(result_error)
}

# 要处理的文件名称
# DISK
# filenames = c("data\\sample\\labeled\\disk_sample_7days_labeled.csv", "data\\sample\\labeled\\disk_sample_2days_labeled.csv")
# outnames = c("arima_error_disk_7DAYS.csv", "arima_error_disk_2DAYS.csv")
# CPU
# filenames = c("data\\sample\\labeled\\CPU_sample_7days_labeled.csv", "data\\sample\\labeled\\CPU_sample_2days_labeled.csv")
# outnames = c("arima_error_CPU_7DAYS.csv", "arima_error_CPU_2DAYS.csv")
# MEM
filenames = c("data\\sample\\labeled\\MEM_sample_7days_labeled.csv", "data\\sample\\labeled\\MEM_sample_2days_labeled.csv")
outnames = c("arima_error_MEM_7DAYS.csv", "arima_error_MEM_2DAYS.csv")
index = 1
index_pre_choose <- c(6, 6 * 6, 6 * 12, 6 * 24, 6 * 48, 6 * 72)
for(filename in filenames)
{
  num_target = 6
  # filename = "data\\sample\\labeled\\sample_7days_labeled.csv"
  print(filename)
  data_in <- read.csv(filename, header = F, as.is = T)
  hostname <- data_in[, 1]
  target <- data_in[, (ncol(data_in) - num_target + 1) : ncol(data_in)]
  datas <- data_in[, 2 : (ncol(data_in) - num_target)]
  results <- NULL
  
  for(line_index in c(1:nrow((datas))))
  {
    print(hostname[line_index])
    line <- datas[line_index, ]
    # 如果不变，则不用预测，而且不变的数据不能建立arima模型
    max_line <- max(line)
    min_line <- min(line)
    if(max_line == min_line)
    {
      pre <- rep(1, pre_length) * max_line
    }else
    {
      pre <- arima_model(line, pre_length)$mean
    }
    result <- c(max(pre[1 : index_pre_choose[1]]), max(pre[1 : index_pre_choose[2]]), max(pre[1 : index_pre_choose[3]]), max(pre[1 : index_pre_choose[4]]), max(pre[1 : index_pre_choose[5]]), max(pre[1 : index_pre_choose[6]]))
    results <- rbind(results, result)
  }
  # write.csv(cbind(hostname, results), "arima_result_disk.csv")
  
  write.csv(cal_error(results, target), outnames[index])
  index = index + 1
}





