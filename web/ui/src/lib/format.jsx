import { Video, Music, Image as ImageIcon, FileText, File } from 'lucide-react';

/*
 * Presentation helpers shared by several components. They live outside any
 * component file so React Fast Refresh can still hot-swap those files — a
 * module that exports both components and plain functions loses that.
 */

const FILE_ICONS = { video: Video, audio: Music, image: ImageIcon, pdf: FileText };

export function fileIcon(type, size = 15) {
  const Icon = FILE_ICONS[type] || File;
  return <Icon size={size} />;
}

/** Seconds -> m:ss, so a hit reads as a place in the file rather than a float. */
export function timecode(seconds) {
  if (seconds === null || seconds === undefined) return null;
  const total = Math.floor(seconds);
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`;
}

/** Where in the source a node sits — a timestamp, a page, or nothing. */
export function locationLabel(node) {
  if (node.start_time !== null && node.start_time !== undefined) return timecode(node.start_time);
  if (node.page_number !== null && node.page_number !== undefined) return `p${node.page_number}`;
  return null;
}

export function formatSize(bytes) {
  if (!bytes) return '—';
  const mb = bytes / 1024 / 1024;
  return mb < 1 ? `${(bytes / 1024).toFixed(0)} KB` : `${mb.toFixed(1)} MB`;
}
