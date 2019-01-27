/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

namespace currender {

enum class LogLevel {
  kOff = 0,
  kDebug = 1,
  kInfo = 2,
  kWarning = 3,
  kError = 4
};

void set_log_level(LogLevel level);
LogLevel get_log_level();
void LOGI(const char *format, ...);
void LOGD(const char *format, ...);
void LOGW(const char *format, ...);
void LOGE(const char *format, ...);

}  // namespace currender
