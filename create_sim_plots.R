library(dplyr)
library(ggplot2)

# struct - year_id, age_start, age_end, measure, draw_0 - n-1

load_and_process_df <- function(fileName, n=200) {
  df <- read.csv(paste("examples/", fileName, sep=""))
  df <- df %>% mutate(
    avgVal = rowMeans(select(., (5:(n+4)))) * 100
  )
  return(df)
}

make_graph <- function(graphNameParams, df1, df2) {
  df1 <- df1 %>% filter(measure == "Blindness")
  df2 <- df2 %>% filter(measure == "Blindness")
  p1 <- ggplot() + 
    geom_line(aes(x=year_id, y=avgVal, color="Original"), data=df1) +
    geom_line(aes(x=year_id, y=avgVal, color="Fixed"), data=df2) +
    scale_y_continuous(breaks=c(0, 5, 10, 15, 20), labels=paste(seq(0, 20, 5), "%", sep=""), limits=c(0, 16)) +
    scale_color_manual(name="Model", values=c("black", "red")) +
    xlab("Years") + 
    ylab("Blindness Prev")
    
  if(graphNameParams[1]=="rebound") {
    p1 <- p1 + 
      scale_x_continuous(breaks=c(100, 125, 140), labels=c("100 (MDA Start)", "125 (MDA Stop)", "200")) +
      ggtitle(paste(graphNameParams[2], "% MFP | 25 years of Annual, 65% Coverage MDA", sep=""))
  } else {
    p1 <- p1 +     
      scale_x_continuous(breaks=c(100, 125, 200), labels=c("100 (MDA Start)", "125", "140(MDA End)")) +
      ggtitle(paste(graphNameParams[2], "% MFP | 40 years of Annual, 65% Coverage MDA", sep=""))
  }
  ggsave(paste(graphNameParams[1], graphNameParams[2], "mfp.png", sep="_"), p1, width=7, height=5)
  return(p1)
}

make_graph_all <- function(graphNameParams, dfs) {
  measureVal = 'Blindness'
  if(length(graphNameParams) > 1) {
    measureVal = graphNameParams[2]
  }
  for(i in 1:length(dfs)) {
    dfs[[i]] <- dfs[[i]] %>% filter(measure == measureVal)
  }
  p1 <- ggplot() + 
    geom_line(aes(x=year_id, y=avgVal, color="Original", linetype="70%"), data=dfs[[1]]) +
    geom_line(aes(x=year_id, y=avgVal, color="Fixed", linetype="70%"), data=dfs[[4]]) +
    
    geom_line(aes(x=year_id, y=avgVal, color="Original", linetype="50%"), data=dfs[[2]]) +
    geom_line(aes(x=year_id, y=avgVal, color="Fixed", linetype="50%"), data=dfs[[5]]) +
    
    geom_line(aes(x=year_id, y=avgVal, color="Original", linetype="30%"), data=dfs[[3]]) +
    geom_line(aes(x=year_id, y=avgVal, color="Fixed", linetype="30%"), data=dfs[[6]]) +
    
    scale_y_continuous(breaks=c(0, 5, 10, 15, 20), labels=paste(seq(0, 20, 5), "%", sep=""), limits=c(0, 16)) +
    scale_color_manual(name="Model", values=c("Original"="black", "Fixed"="red"), breaks=c("Original", "Fixed")) +
    scale_linetype_manual(name="MFP", values=c("70%"="solid", "50%"="dashed", "30%"="dotdash"), breaks=c("70%", "50%", "30%")) +
    xlab("Years") + 
    ylab("Blindness Prev")
  
  if(graphNameParams[1]=="rebound") {
    p1 <- p1 + 
      scale_x_continuous(breaks=c(100, 125, 140), labels=c("100 (MDA Start)", "125 (MDA Stop)", "140")) +
      ggtitle(paste(graphNameParams[2], "25 years of Annual, 65% Coverage MDA | Varying MFP", sep=""))
  } else {
    p1 <- p1 +     
      scale_x_continuous(breaks=c(100, 125, 140), labels=c("100 (MDA Start)", "125", "140(MDA End)")) +
      ggtitle(paste(measure, "40 years of Annual, 65% Coverage MDA | Varying MFP", sep=""))
  }
  #ggsave(paste(graphNameParams[1], "varying_mfp.png", sep="_"), p1, width=7, height=5)
  return(p1)
}

no_fix_70 <- load_and_process_df("70_mfp_no_rebound_dynamics.csv")
fix_70 <- load_and_process_df("70_mfp_no_rebound_dynamics_fixed.csv")
fix_70_2 <- load_and_process_df("70_mfp_no_rebound_dynamics_fixed_2.csv")
reb_fix_70 <- load_and_process_df("70_mfp_rebound_dynamics_fixed.csv")
make_graph(c("no_rebound", 70), no_fix_70, fix_70)
make_graph(c("no_rebound", 70), no_fix_70, fix_70_2)
make_graph(c("no_rebound", 70), fix_70, fix_70_2)


no_fix_50 <- load_and_process_df("50_mfp_no_rebound_dynamics.csv")
fix_50 <- load_and_process_df("50_mfp_no_rebound_dynamics_fixed.csv")
fix_50_2 <- load_and_process_df("50_mfp_no_rebound_dynamics_fixed_2.csv")
reb_fix_50 <- load_and_process_df("50_mfp_rebound_dynamics_fixed.csv", n=10)
make_graph(c("no_rebound", 50), no_fix_50, fix_50)
make_graph(c("no_rebound", 50), no_fix_50, fix_50_2)
make_graph(c("no_rebound", 50), fix_50, fix_50_2)


no_fix_30 <- load_and_process_df("30_mfp_no_rebound_dynamics.csv")
fix_30 <- load_and_process_df("30_mfp_no_rebound_dynamics_fixed.csv")
fix_30_2 <- load_and_process_df("30_mfp_no_rebound_dynamics_fixed_2.csv")
reb_fix_30 <- load_and_process_df("30_mfp_rebound_dynamics_fixed.csv")
make_graph(c("no_rebound", 30), no_fix_30, fix_30)

make_graph_all(c("no_rebound"), list(no_fix_70, no_fix_50, no_fix_30, fix_70, fix_50, fix_30))

for (measure in unique(fix_30$measure)) {
  print(make_graph_all(c("no_rebound", measure), list(no_fix_70, no_fix_50, no_fix_30, fix_70_2, fix_50_2, fix_30_2)))
}



make_graph_all(c("no_rebound"), list(fix_70, fix_50, fix_30, fix_70_2, fix_50_2, fix_30_2))


make_graph_all(c("rebound"), list(no_fix_70, no_fix_50, no_fix_30, reb_fix_70, reb_fix_70, reb_fix_70))


